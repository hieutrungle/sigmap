import time
import os
import glob
import re
import subprocess
import argparse

gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sigmap.drl.agents.soft_actor_critic import SoftActorCritic
from sigmap.drl.infrastructure.replay_buffer import WirelessReplayBuffer

# import sigmap.drl.env_configs

import os
import time

import gymnasium as gym
from gymnasium import wrappers
import numpy as np
import torch
from sigmap.drl.infrastructure import pytorch_utils as ptu
import tqdm

from sigmap.drl.infrastructure.logger import TensorboardLogger

from utils import scripting_utils, utils, timer

import argparse
from sigmap.drl.envs import register_envs

register_envs()


def run_training_loop(
    drl_config: dict,
    # sionna_config: scripting_utils.Config,
    tsb_logger: TensorboardLogger,
    args: argparse.Namespace,
):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    if args.verbose:
        utils.log_args(args)
        # utils.log_config(sionna_config)
        utils.log_config(drl_config)

    env = drl_config.make_env()
    # eval_env = drl_config.make_env()
    ep_len = drl_config.ep_len or env.spec.max_episode_steps
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert (
        not discrete
    ), "Our wireless DRL implementation only supports continuous action spaces."

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    # ob_shape = env.observation_space["focal_pts"].shape
    # ac_shape = env.action_space.shape
    # ac_dim = np.prod(ac_shape)

    # # initialize agent
    # agent = SoftActorCritic(
    #     ob_shape,
    #     ac_dim,
    #     **drl_config.agent_kwargs,
    # )

    ob_space = env.observation_space
    ob_shapes = []
    for key in ob_space.keys():
        ob_shapes.append(ob_space[key].shape)
    ob_shapes = tuple(ob_shapes)
    ac_space = env.action_space
    ac_shape = ac_space.shape
    # TODO: implement SAC with ob_space and ac_dim
    agent = SoftActorCritic(
        ob_shapes,
        ac_shape,
        **drl_config.agent_kwargs,
    )

    assets_dir = utils.get_asset_dir()
    replay_buffer_dir = os.path.join(assets_dir, "replay_buffer")
    buffer_name = drl_config.log_name + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    buffer_saved_dir = os.path.join(replay_buffer_dir, buffer_name)
    utils.mkdir_not_exists(buffer_saved_dir)
    replay_buffer = WirelessReplayBuffer(
        drl_config.replay_buffer_capacity, buffer_saved_dir
    )

    (observation, info) = env.reset()

    for step in tqdm.trange(drl_config.total_steps, dynamic_ncols=True):

        # accumulate data in replay buffer
        if step < drl_config.random_steps:
            action = env.action_space.sample()
        else:
            # TODO: get correct action from agent
            action = agent.get_action(observation)

        # with timer.Timer(
        #     text="Elapsed env step time: {:0.4f} seconds\n", logger_fn=utils.logger.log
        # ):
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        done = done or (info.get("episode", {}).get("l", 0) >= ep_len)
        reward = np.array(reward, dtype=np.float32)
        done = np.array(done, dtype=np.float32)
        replay_buffer.insert(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done,
        )
        if done:
            tsb_logger.log_scalar(info["episode"]["r"], "train_return", step)
            tsb_logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
            observation, info = env.reset()
        else:
            observation = next_observation

        # train agent
        if step > drl_config.training_starts:
            # TODO: fix batch sampling
            batch = replay_buffer.sample(drl_config.batch_size)
            obs, actions, rewards, next_obs, dones = batch

            # TODO: implement update method in SAC
            update_info = agent.update(obs, actions, rewards, next_obs, dones)

            # logging
            update_info["actor_lr"] = agent.actor_lr_scheduler.get_last_lr()[0]
            update_info["critic_lr"] = agent.critics_lr_scheduler.get_last_lr()[0]

            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    tsb_logger.log_scalar(v, k, step)
                tsb_logger.flush()

        if step == 10:
            break
    print(f"len of replay buffer: {len(replay_buffer)}")
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drl_config_file", "-dcfg", type=str, required=True)
    parser.add_argument("--sionna_config_file", "-scfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)
    parser.add_argument("--log_interval", type=int, default=1000)

    parser.add_argument("--verbose", "-v", action="store_true")

    # Wireless environment specific arguments
    parser.add_argument("--num_devices", "-ndev", type=int, default=1)
    parser.add_argument("--num_tiles_per_device", "-ntiles", type=int, default=70)
    parser.add_argument("--controlled_elements", "-ce", type=int, default=2)

    args = parser.parse_args()

    # sionna_config = scripting_utils.make_sionna_config(args.sionna_config_file)
    drl_config = scripting_utils.make_drl_config(args.drl_config_file, args)
    tsb_logger = scripting_utils.make_tensorboard_logger(drl_config)

    run_training_loop(drl_config, tsb_logger, args)


if __name__ == "__main__":
    main()
