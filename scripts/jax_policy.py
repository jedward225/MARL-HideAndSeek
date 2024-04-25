import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn
from flax.core import FrozenDict

import argparse
from functools import partial
import math

import madrona_learn
from madrona_learn import (
    Policy, ActorCritic, BackboneShared, BackboneSeparate,
    BackboneEncoder, RecurrentBackboneEncoder,
    ObservationsEMANormalizer, ObservationsCaster,
)

from madrona_learn.models import (
    LayerNorm,
    MLP,
    EntitySelfAttentionNet,
    DenseLayerDiscreteActor,
    DenseLayerCritic,
)
from madrona_learn.rnn import LSTM

def assert_valid_input(tensor):
    checkify.check(jnp.isnan(tensor).any() == False, "NaN!")
    checkify.check(jnp.isinf(tensor).any() == False, "Inf!")

def assert_valid_input(tensor):
    return None
    #checkify.check(jnp.isnan(tensor).any() == False, "NaN!")
    #checkify.check(jnp.isinf(tensor).any() == False, "Inf!")

class PolicyRNN(nn.Module):
    rnn: nn.Module
    norm: nn.Module

    @staticmethod
    def create(num_hidden_channels, num_layers, dtype, rnn_cls = LSTM):
        return PolicyRNN(
            rnn = rnn_cls(
                num_hidden_channels = num_hidden_channels,
                num_layers = num_layers,
                dtype = dtype,
            ),
            norm = LayerNorm(dtype=dtype),
        )

    @nn.nowrap
    def init_recurrent_state(self, N):
        return self.rnn.init_recurrent_state(N)

    @nn.nowrap
    def clear_recurrent_state(self, rnn_states, should_clear):
        return self.rnn.clear_recurrent_state(rnn_states, should_clear)

    def setup(self):
        pass

    def __call__(
        self,
        cur_hiddens,
        x,
        train,
    ):
        out, new_hiddens = self.rnn(cur_hiddens, x, train)
        return self.norm(out), new_hiddens

    def sequence(
        self,
        start_hiddens,
        seq_ends,
        seq_x,
        train,
    ):
        return self.norm(
            self.rnn.sequence(start_hiddens, seq_ends, seq_x, train))

def extract_self_obs(obs):
    obs, prep_counter = obs.pop('prep_counter')
    obs, self_data = obs.pop('self_data')
    obs, self_type = obs.pop('self_type')
    obs, self_mask = obs.pop('self_mask')
    obs, agent_lidar = obs.pop('self_lidar')

    self_ob = jnp.concatenate([
        prep_counter,
        self_data,
        self_type,
        agent_lidar,
    ], axis=-1)

    return obs, self_ob

class PrefixCommon(nn.Module):
    dtype: jnp.dtype

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        jax.tree_map(lambda x: assert_valid_input(x), obs)
        return obs


class SimpleNet(nn.Module):
    dtype: jnp.dtype

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        num_batch_dims = len(obs['self'].shape) - 1
        obs = jax.tree_map(
            lambda o: o.reshape(*o.shape[0:num_batch_dims], -1), obs)

        flattened, _ = jax.tree_util.tree_flatten(obs)
        flattened = jnp.concatenate(flattened, axis=-1)

        return MLP(
                num_channels = 256,
                num_layers = 3,
                dtype = self.dtype,
            )(flattened, train)

class ActorNet(nn.Module):
    dtype: jnp.dtype
    use_simple: bool

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        obs, self_ob = extract_self_obs(obs)
        
        obs, agent_data = obs.pop('agent_data')
        obs, box_data = obs.pop('box_data')
        obs, ramp_data = obs.pop('ramp_data')
        obs, vis_agents_mask = obs.pop('vis_agents_mask')
        obs, vis_boxes_mask = obs.pop('vis_boxes_mask')
        obs, vis_ramps_mask = obs.pop('vis_ramps_mask')

        assert len(obs) == 0

        agent_data = agent_data * vis_agents_mask
        box_data = box_data * vis_boxes_mask
        ramp_data = ramp_data * vis_ramps_mask

        obs = FrozenDict({
            'self': self_ob,
            'agents': agent_data, 
            'boxes': box_data, 
            'ramps': ramp_data, 
        })

        if self.use_simple:
            return SimpleNet(dtype=self.dtype)(obs, train)
        else:
            return EntitySelfAttentionNet(
                    num_embed_channels = 128,
                    num_out_channels = 256,
                    num_heads = 4,
                    dtype = self.dtype,
                )(obs, train=train)


class CriticNet(nn.Module):
    dtype: jnp.dtype
    use_simple: bool

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        obs, self_ob = extract_self_obs(obs)
        
        obs, agent_data = obs.pop('agent_data')
        obs, box_data = obs.pop('box_data')
        obs, ramp_data = obs.pop('ramp_data')
        obs, vis_agents_mask = obs.pop('vis_agents_mask')
        obs, vis_boxes_mask = obs.pop('vis_boxes_mask')
        obs, vis_ramps_mask = obs.pop('vis_ramps_mask')

        assert len(obs) == 0

        obs = FrozenDict({
            'self': self_ob,
            'agents': agent_data, 
            'boxes': box_data, 
            'ramps': ramp_data, 
        })

        if self.use_simple:
            return SimpleNet(dtype=self.dtype)(obs, train)
        else:
            return EntitySelfAttentionNet(
                    num_embed_channels = 128,
                    num_out_channels = 256,
                    num_heads = 4,
                    dtype = self.dtype,
                )(obs, train=train)

def make_policy(dtype):
    actor_encoder = RecurrentBackboneEncoder(
        net = ActorNet(dtype, use_simple=False),
        rnn = PolicyRNN.create(
            num_hidden_channels = 256,
            num_layers = 1,
            dtype = dtype,
        ),
    )

    critic_encoder = RecurrentBackboneEncoder(
        net = CriticNet(dtype, use_simple=False),
        rnn = PolicyRNN.create(
            num_hidden_channels = 256,
            num_layers = 1,
            dtype = dtype,
        ),
    )

    backbone = BackboneSeparate(
        prefix = PrefixCommon(
            dtype = dtype,
        ),
        actor_encoder = actor_encoder,
        critic_encoder = critic_encoder,
    )

    actor_critic = ActorCritic(
        backbone = backbone,
        actor = DenseLayerDiscreteActor(
            actions_num_buckets = [4, 8, 5, 2],
            dtype = dtype,
        ),
        critic = DenseLayerCritic(dtype=dtype),
    )

    obs_preprocess = ObservationsEMANormalizer.create(
        decay = 0.99999,
        dtype = dtype,
        prep_fns = {
            'prep_counter': lambda x: (x.astype(jnp.float32) / 96).astype(dtype),
            'self_type': lambda x: x.astype(dtype),
            'vis_agents_mask': lambda x: x.astype(dtype),
            'vis_boxes_mask': lambda x: x.astype(dtype),
            'vis_ramps_mask': lambda x: x.astype(dtype),
        },
        skip_normalization = {
            'prep_counter',
            'self_type',
            'self_mask',
            'vis_agents_mask',
            'vis_boxes_mask',
            'vis_ramps_mask',
        },
    )

    def get_episode_scores(episode_result):
        return episode_result

    return Policy(
        actor_critic = actor_critic,
        obs_preprocess = obs_preprocess,
        get_episode_scores = get_episode_scores,
    )

    return policy
