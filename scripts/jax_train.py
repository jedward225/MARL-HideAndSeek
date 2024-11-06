import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn

import argparse
from functools import partial
from time import time
from dataclasses import dataclass
import os
import numpy as np

import gpu_hideseek
from gpu_hideseek import SimFlags
from gpu_hideseek.madrona import ExecMode

import madrona_learn
from madrona_learn import (
    ActionsConfig, TrainConfig, TrainHooks, PPOConfig, PBTConfig, ParamExplore,
    TensorboardWriter, WandbWriter
)
import wandb

from jax_policy import make_policy
from common import print_elos

madrona_learn.cfg_jax_mem(0.8)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--ckpt-dir', type=str, required=True)
arg_parser.add_argument('--tb-dir', type=str, required=True)
arg_parser.add_argument('--run-name', type=str, required=True)
arg_parser.add_argument('--restore', type=int)

arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-updates', type=int, required=True)
arg_parser.add_argument('--steps-per-update', type=int, default=40)
arg_parser.add_argument('--num-bptt-chunks', type=int, default=8)
arg_parser.add_argument('--num-minibatches', type=int, default=2)
arg_parser.add_argument('--num-epochs', type=int, default=4)

arg_parser.add_argument('--lr', type=float, default=1e-4)
arg_parser.add_argument('--gamma', type=float, default=0.998)
arg_parser.add_argument('--entropy-loss-coef', type=float, default=0.01)
arg_parser.add_argument('--value-loss-coef', type=float, default=1.0)
arg_parser.add_argument('--clip-value-loss', action='store_true')

arg_parser.add_argument('--num-channels', type=int, default=256)
arg_parser.add_argument('--separate-value', action='store_true')
arg_parser.add_argument('--fp16', action='store_true')
arg_parser.add_argument('--bf16', action='store_true')

arg_parser.add_argument('--pbt-ensemble-size', type=int, default=0)
arg_parser.add_argument('--pbt-past-policies', type=int, default=0)

arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--profile-port', type=int, default=None)
arg_parser.add_argument('--wandb', action='store_true')

arg_parser.add_argument('--num-hiders', type=int, default=3)
arg_parser.add_argument('--num-seekers', type=int, default=3)

arg_parser.add_argument('--eval-frequency', type=int, default=500)

args = arg_parser.parse_args()

sim = gpu_hideseek.HideAndSeekSimulator(
    exec_mode = ExecMode.CUDA if args.gpu_sim else ExecMode.CPU,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    sim_flags = SimFlags.RandomFlipTeams | SimFlags.UseFixedWorld | SimFlags.ZeroAgentVelocity,
    #sim_flags = SimFlags.RandomFlipTeams | SimFlags.ZeroAgentVelocity,
    min_hiders = args.num_hiders,
    max_hiders = args.num_hiders,
    min_seekers = args.num_seekers,
    max_seekers = args.num_seekers,
    num_pbt_policies = args.pbt_ensemble_size,
    rand_seed = 5,
)

jax_gpu = jax.devices()[0].platform == 'gpu'

sim_fns = sim.jax(jax_gpu)
if args.wandb:
    tb_writer = WandbWriter(os.path.join(args.tb_dir, args.run_name), args=args)
else:
    tb_writer = TensorboardWriter(os.path.join(args.tb_dir, args.run_name))

@dataclass(frozen=True)
class HideSeekHooks(TrainHooks):
    pass

dev = jax.devices()[0]

if args.pbt_ensemble_size != 0:
    assert args.num_hiders == args.num_seekers

    pbt_cfg = PBTConfig(
        num_teams = 2,
        team_size = args.num_hiders,
        num_train_policies = args.pbt_ensemble_size,
        num_past_policies = args.pbt_past_policies,
        #self_play_portion = 0.875,
        #cross_play_portion = 0.0,
        #past_play_portion = 0.125,
        self_play_portion = 0.0,
        cross_play_portion = 0.0,
        past_play_portion = 1.0,
        reward_hyper_params_explore = {},
    )
else:
    pbt_cfg = None

if args.fp16:
    dtype = jnp.float16
elif args.bf16:
    dtype = jnp.bfloat16
else:
    dtype = jnp.float32


if pbt_cfg:
    lr = ParamExplore(
        base = args.lr,
        min_scale = 0.1,
        max_scale = 10.0,
        log10_scale = True,
    )

    entropy_coef = ParamExplore(
        base = args.entropy_loss_coef,
        min_scale = 0.1,
        max_scale = 10.0,
        log10_scale = True,
    )
else:
    lr = args.lr
    entropy_coef = args.entropy_loss_coef

cfg = TrainConfig(
    num_worlds = args.num_worlds,
    num_agents_per_world = args.num_hiders + args.num_seekers,
    num_updates = args.num_updates,
    actions = ActionsConfig(
        actions_num_buckets = [ 5, 5, 5, 2, 2 ],
    ),
    steps_per_update = args.steps_per_update,
    num_bptt_chunks = args.num_bptt_chunks,
    lr = lr,
    gamma = args.gamma,
    gae_lambda = 0.95,
    algo = PPOConfig(
        num_mini_batches = args.num_minibatches,
        clip_coef = 0.2,
        value_loss_coef = args.value_loss_coef,
        entropy_coef = entropy_coef,
        max_grad_norm = 5,
        num_epochs = args.num_epochs,
        clip_value_loss = args.clip_value_loss,
    ),
    pbt = pbt_cfg,
    dreamer_v3_critic = True,
    #value_normalizer_decay = 0.999,
    compute_dtype = dtype,
    seed = 5,
    metrics_buffer_size = 10,
)

policy = make_policy(dtype, cfg.actions)

if args.restore:
    restore_ckpt = os.path.join(
        args.ckpt_dir, args.run_name, str(args.restore))
else:
    restore_ckpt = None

last_time = 0
last_update = 0

def _log_metrics_host_cb(training_mgr):
    global last_time, last_update

    update_id = int(training_mgr.update_idx)

    cur_time = time()
    update_diff = update_id - last_update

    print(f"Update: {update_id}")
    if last_time != 0:
        print("  FPS:", args.num_worlds * args.steps_per_update * update_diff / (cur_time - last_time))

    last_time = cur_time
    last_update = update_id

    #metrics.pretty_print()

    if args.pbt_ensemble_size > 0:
        old_printopts = np.get_printoptions()
        np.set_printoptions(formatter={'float_kind':'{:.1e}'.format}, linewidth=150)

        lrs = np.asarray(training_mgr.state.train_states.hyper_params.lr)
        entropy_coefs = np.asarray(
            training_mgr.state.train_states.hyper_params.entropy_coef)

        elos = np.asarray(training_mgr.state.policy_states.mmr.elo)
        print_elos(elos)

        np.set_printoptions(**old_printopts)

        print()

        for i in range(elos.shape[0]):
            tb_writer.scalar(f"p{i}/elo", elos[i], update_id)

        num_train_policies = lrs.shape[0]
        for i in range(lrs.shape[0]):
            tb_writer.scalar(f"p{i}/lr", lrs[i], update_id)
            tb_writer.scalar(f"p{i}/entropy_coef", entropy_coefs[i], update_id)

    training_mgr.log_metrics_tensorboard(tb_writer)

    return ()


def update_loop(training_mgr):
    assert args.eval_frequency % 10 == 0

    def inner_iter(i, training_mgr):
        return training_mgr.update_iter()

    def outer_iter(i, training_mgr):
        training_mgr = lax.fori_loop(0, 10, inner_iter, training_mgr)

        jax.experimental.io_callback(
            _log_metrics_host_cb, (), training_mgr, ordered=True)

        return training_mgr

    return lax.fori_loop(0, args.eval_frequency // 10, outer_iter, training_mgr)

def eval_elo(training_mgr):
    return madrona_learn.eval_elo(training_mgr)

def train():
    global last_time 

    training_mgr = madrona_learn.init_training(dev, cfg, sim_fns, policy,
        restore_ckpt=restore_ckpt,
        profile_port=args.profile_port)

    assert training_mgr.update_idx % args.eval_frequency == 0
    num_outer_iters = ((args.num_updates - int(training_mgr.update_idx)) //
        args.eval_frequency)

    update_loop_compiled = madrona_learn.aot_compile(update_loop, training_mgr)

    eval_elo_compiled = madrona_learn.aot_compile(eval_elo, training_mgr)

    last_time = time()

    for i in range(num_outer_iters):
        err, training_mgr = update_loop_compiled(training_mgr)
        err.throw()

        err, training_mgr = eval_elo_compiled(training_mgr)
        err.throw()

        print(training_mgr.state.policy_states.mmr.elo)

        err, training_mgr = eval_elo_compiled(training_mgr)
        err.throw()

        print(training_mgr.state.policy_states.mmr.elo)

        training_mgr.save_ckpt(f"{args.ckpt_dir}/{args.run_name}")
    
    madrona_learn.stop_training(training_mgr)

if __name__ == "__main__":
    try:
        train()
    except:
        tb_writer.flush()
        raise
    
    tb_writer.flush()
    del sim
