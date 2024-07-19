import time
import os
import argparse

# gpu_num = 0
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".10"

import tensorflow as tf

devices = tf.config.list_physical_devices("GPU")
print(f"number of GPUs: {len(devices)}")

from sigmap.drl.agents.wireless_sac_jax import SoftActorCritic
from sigmap.drl.infrastructure.replay_buffer import WirelessReplayBuffer

import gymnasium as gym
import numpy as np
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
    env: gym.Env,
    agent: SoftActorCritic,
    replay_buffer: WirelessReplayBuffer,
):
    ep_len = drl_config.ep_len or env.spec.max_episode_steps

    best_return = -np.inf
    (observation, info) = env.reset()

    try:
        # load agent if exists, resume training
        agent = agent.load()
        print(f"Resuming training from step {agent.checkpoint_manager.latest_step()}")
        start_step = agent.checkpoint_manager.latest_step()
    except Exception as e:
        print(f"Error in loading agent: {e}")
        print(f"Training from scratch")
        start_step = 0
    start_step = int(start_step)

    for step in tqdm.tqdm(
        range(start_step, drl_config.total_steps),
        total=drl_config.total_steps,
        dynamic_ncols=True,
        initial=start_step,
    ):
        # accumulate data in replay buffer
        if step < drl_config.random_steps * 1 / 2:
            observation, info = env.reset()
            action = env.action_space.sample()
        elif step < drl_config.random_steps:
            action = env.action_space.sample()
        else:
            action = agent.get_action(observation)

        # with timer.Timer(
        #     text="Elapsed env step time: {:0.4f} seconds\n", logger_fn=utils.logger.log
        # ):
        # ! TODO: GPU memory leak in env because of tensorflow persistent state
        # ! may use subprocess to run env in separate process
        # ! but this makes it difficult to debug and increase run time
        try:
            next_observation, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            print(f"Error in step {step}: {e}")
            time.sleep(2)
            continue

        if terminated:
            actions = next_observation["focal_pts"] - observation["focal_pts"]
            print(f"Terminated at step {step}")
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
        train_return = info["episode"]["r"]
        tsb_logger.log_scalar(train_return, "train_return", step)
        tsb_logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
        if done:
            observation, info = env.reset()
        else:
            observation = next_observation

        # train agent
        if step > drl_config.training_starts:
            for _ in range(drl_config.num_train_steps_per_env_step):
                batch = replay_buffer.sample(drl_config.batch_size)
                obs, actions, rewards, next_obs, dones = (
                    batch["observations"],
                    batch["actions"],
                    batch["rewards"],
                    batch["next_observations"],
                    batch["dones"],
                )
                update_info = agent.update(obs, actions, rewards, next_obs, dones, step)

            # logging

            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    tsb_logger.log_scalar(v, k, step)
                tsb_logger.flush()

            if train_return > best_return:
                best_return = train_return
                agent.save(step)
    agent.wait_for_checkpoint()
    return


def run_eval_loop(
    drl_config: dict,
    tsb_logger: TensorboardLogger,
    args: argparse.Namespace,
    env,
    agent: SoftActorCritic,
):
    import matplotlib.pyplot as plt

    ep_len = drl_config.ep_len or env.spec.max_episode_steps
    ep_len = min(ep_len, 100)
    env.eval()
    agent = agent.load()

    eval_sums = np.zeros(ep_len)
    eval_mins = np.ones(ep_len) * np.inf
    eval_maxs = np.ones(ep_len) * -np.inf
    max_step = 0
    num_evals = 3
    eval_count = np.zeros(ep_len)
    eval_traj = np.zeros(ep_len)

    for i in range(num_evals):
        (observation, info) = env.reset()
        for step in tqdm.trange(ep_len, dynamic_ncols=True):
            action = agent.get_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            eval_return = info["episode"]["r"]
            eval_traj[step] = eval_return
            eval_count[step] += 1
            eval_sums[step] += eval_return
            max_step = max(max_step, step)

            tsb_logger.log_scalar(eval_return, f"eval_return_{i}", step)

            if done:
                print(f"current position: {observation['focal_pts']}")
                print(f"Terminated at step {step}")
                break
            else:
                observation = next_observation

        # the current max step for indexing
        t = max_step + 1
        eval_mins[:t] = np.minimum(eval_mins[:t], eval_traj[:t])
        eval_maxs[:t] = np.maximum(eval_maxs[:t], eval_traj[:t])

    max_step = max_step + 1
    eval_means = eval_sums[:max_step] / eval_count[:max_step]

    # trim the arrays to the max step
    eval_mins = eval_mins[:max_step]
    eval_maxs = eval_maxs[:max_step]

    # log the evaluation results to tensorboard
    for step in range(max_step):
        tsb_logger.log_scalar(eval_means[step], "eval_return_mean", step)
        tsb_logger.log_scalar(eval_mins[step], "eval_return_min", step)
        tsb_logger.log_scalar(eval_maxs[step], "eval_return_max", step)
        tsb_logger.log_scalar(info["episode"]["l"], "eval_ep_len", step)

    # plot the evaluation results
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.plot(eval_means, label="mean")
    ax.fill_between(
        range(max_step), eval_mins, eval_maxs, alpha=0.3, label="min-max range"
    )
    ax.plot(eval_mins, label="min", linestyle="--")
    ax.plot(eval_maxs, label="max", linestyle="--")
    ax.grid()
    ax.legend()
    ax.set_title("Evaluation Results")
    ax.set_xlabel("steps")
    ax.set_ylabel("return")
    # save the plot
    saved_path = drl_config.saved_path
    fig_name = f"eval_results.png"
    fig_path = os.path.join(saved_path, fig_name)
    plt.savefig(fig_path)


def main():

    args = parse_agrs()
    drl_config = scripting_utils.make_drl_config(args.drl_config_file, args)
    tsb_logger = scripting_utils.make_tensorboard_logger(drl_config)

    # set random seeds
    np.random.seed(args.seed)

    if args.verbose:
        utils.log_args(args)
        utils.log_config(drl_config)

    env = drl_config.make_env()
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert (
        not discrete
    ), "Our wireless DRL implementation only supports continuous action spaces."

    ob_space = env.observation_space
    ob_shapes = {}
    for k, v in ob_space.spaces.items():
        ob_shapes[k] = v.shape
    ac_space = env.action_space
    ac_shape = ac_space.shape

    seed = drl_config.seed

    agent = SoftActorCritic.create(
        ob_shapes,
        ac_shape,
        **drl_config.agent_kwargs,
    )

    args.command = str(args.command).lower()
    if args.command == "train":

        assets_dir = utils.get_asset_dir()
        replay_buffer_dir = os.path.join(assets_dir, "replay_buffer")
        buffer_saved_dir = os.path.join(replay_buffer_dir, drl_config.log_name)
        utils.mkdir_not_exists(buffer_saved_dir)
        replay_buffer = WirelessReplayBuffer(
            drl_config.replay_buffer_capacity, buffer_saved_dir, seed=seed
        )

        run_training_loop(drl_config, tsb_logger, args, env, agent, replay_buffer)
    elif args.command == "eval":
        run_eval_loop(
            drl_config,
            tsb_logger,
            args,
            env,
            agent,
        )
    else:
        raise ValueError(f"Invalid command: {args.command}")


def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drl_config_file", "-dcfg", type=str, required=True)
    parser.add_argument("--sionna_config_file", "-scfg", type=str, required=True)

    parser.add_argument("--command", "-cmd", type=str, required=True)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=1)

    parser.add_argument("--verbose", "-v", action="store_true")

    # Wireless environment specific arguments
    parser.add_argument("--num_devices", "-ndev", type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
