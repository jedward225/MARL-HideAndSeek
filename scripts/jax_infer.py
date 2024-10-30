import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn
import numpy as np

import argparse
from functools import partial

import gpu_hideseek
from gpu_hideseek import SimFlags
from gpu_hideseek.madrona import ExecMode

import madrona_learn

from jax_policy import make_policy
from common import print_elos

madrona_learn.init(0.5)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-steps', type=int, default=200)

arg_parser.add_argument('--ckpt-path', type=str, required=True)
arg_parser.add_argument('--single-policy', type=int, default=None)
arg_parser.add_argument('--record-log', type=str)

arg_parser.add_argument('--print-obs', action='store_true')
arg_parser.add_argument('--print-action-probs', action='store_true')
arg_parser.add_argument('--print-rewards', action='store_true')

arg_parser.add_argument('--num-hiders', type=int)
arg_parser.add_argument('--num-seekers', type=int)

arg_parser.add_argument('--fp16', action='store_true')
arg_parser.add_argument('--bf16', action='store_true')
arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--gpu-id', type=int, default=0)

args = arg_parser.parse_args()

dev = jax.devices()[0]

if args.fp16:
    dtype = jnp.float16
elif args.bf16:
    dtype = jnp.bfloat16
else:
    dtype = jnp.float32

policy = make_policy(dtype)

if args.single_policy != None:
    policy_states, num_policies = madrona_learn.eval_load_ckpt(
        policy, args.ckpt_path, single_policy = args.single_policy)
else:
    policy_states, num_policies = madrona_learn.eval_load_ckpt(
        policy, args.ckpt_path, train_only=True)

sim = gpu_hideseek.HideAndSeekSimulator(
    exec_mode = ExecMode.CUDA if args.gpu_sim else ExecMode.CPU,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    num_pbt_policies = num_policies if num_policies > 1 else 1,
    rand_seed = 5,
    #sim_flags = SimFlags.RandomFlipTeams | SimFlags.UseFixedWorld,
    #sim_flags = SimFlags.UseFixedWorld | SimFlags.ZeroAgentVelocity,
    sim_flags = SimFlags.ZeroAgentVelocity,
    min_hiders = args.num_hiders,
    max_hiders = args.num_hiders,
    min_seekers = args.num_seekers,
    max_seekers = args.num_seekers,
)

ckpt_tensor = sim.ckpt_tensor()
action_tensor = sim.action_tensor()

assert args.num_hiders == args.num_seekers

team_size = args.num_hiders
num_teams = 2

num_agents_per_world = team_size * num_teams

jax_gpu = jax.devices()[0].platform == 'gpu'

sim_fns = sim.jax(jax_gpu)

if args.record_log:
    record_log_file = open(args.record_log, 'wb')
else:
    record_log_file = None

step_idx = 0

def host_cb(obs, actions, action_probs, critic, dones, rewards, ckpts):
    global step_idx

    if args.print_obs:
        print(obs)

    print(f"\nStep {step_idx}")

    if args.print_action_probs:
        for i in range(actions.shape[0]):
            if i % num_agents_per_world == 0:
                print(f"World {i // num_agents_per_world}")

            print(f" Agent {i % num_agents_per_world}:")
            print("  Action:", actions[..., i, :])

            print(f"  Move Amount Probs: {float(action_probs[0][i][0]):.2e} {float(action_probs[0][i][1]):.2e} {float(action_probs[0][i][2]):.2e}")
            print(f"  Turn Probs:        {float(action_probs[1][i][0]):.2e} {float(action_probs[1][i][1]):.2e} {float(action_probs[1][i][2]):.2e}")

    if args.print_rewards:
        print("Rewards:", rewards)


    np.asarray(jax.device_get(ckpts)).tofile(record_log_file)

    step_idx += 1

    return ()

def iter_cb(step_data):
    cb = partial(jax.experimental.io_callback, host_cb, ())

    sim_state = step_data['sim_state']

    if args.record_log:
        save_ckpts_out = sim_fns['save_ckpts']({
            'state': sim_state,
            'should_save': jnp.ones((args.num_worlds, 1), jnp.int32),
        })

        sim_state = save_ckpts_out['state']
        ckpts = save_ckpts_out['ckpts']
    else:
        ckpts = None

    cb(step_data['obs'],
       step_data['actions'],
       step_data['action_probs'],
       step_data['critic'],
       step_data['dones'],
       step_data['rewards'],
       ckpts)

cfg = madrona_learn.EvalConfig(
    num_worlds = args.num_worlds,
    team_size = team_size,
    num_teams = num_teams,
    num_eval_steps = args.num_steps,
    reward_gamma = 0.998,
    eval_competitive = num_policies > 1,
    policy_dtype = dtype,
)

if num_policies > 1:
    mmrs = policy_states.mmr
    print_elos(mmrs.elo)

mmrs = madrona_learn.eval_policies(
    dev, cfg, sim_fns, policy, policy_states, iter_cb)

if num_policies > 1:
    print_elos(mmrs.elo)

del sim
