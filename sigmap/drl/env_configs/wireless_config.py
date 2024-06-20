from typing import Tuple, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
from sigmap.drl.networks.mlp_policy import MLPPolicy
from sigmap.drl.networks.state_action_value_critic import StateActionCritic

import gymnasium as gym
from sigmap.drl.env_configs.schedule import CosineAnnealingWarmupRestarts
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

import argparse


def wireless_config(
    args: argparse.Namespace,
    env_name: str,
    exp_name: Optional[str] = None,
    replay_buffer_capacity: int = 1000000,
    total_steps: int = 300000,
    random_steps: int = 5000,
    training_starts: int = 10000,
    ep_len: Optional[int] = None,
    # Training settings
    actor_learning_rate: float = 3e-4,
    critic_learning_rate: float = 3e-4,
    alpha_learning_rate: float = 3e-4,
    batch_size: int = 128,
    num_train_steps_per_env_step: int = 1,
    # Actor-critic configuration
    hidden_sizes: Sequence[int] = [128, 128, 128],
    discount: float = 0.99,
    ema_decay: float = 0.995,  # soft target update rate
    num_critics: int = 2,
    num_critic_updates: int = 5,
    temperature: float = 0.05,  # temperature for entropy
    num_actor_samples: int = 5,
    saved_path: Optional[str] = None,
    seed: int = 0,
):

    def make_lr_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        max_lr = (actor_learning_rate + critic_learning_rate) / 2
        min_lr = max_lr / 50
        return CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=int(
                ((total_steps - training_starts) * num_train_steps_per_env_step) // 1
            ),
            cycle_mult=1.0,
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_steps=int(
                ((total_steps - training_starts) * num_train_steps_per_env_step) // 16
            ),
            gamma=1 / 2,
        )

    def make_env(render: bool = False):
        return gym.make(
            env_name,
            sionna_config_file=args.sionna_config_file,
            num_devices=args.num_devices,
            num_tiles_per_device=args.num_tiles_per_device,
            controlled_elements=args.controlled_elements,
            render_mode="rgb_array" if render else None,
        )

    log_string = "{}_{}_s{}_aclr{}_crlr{}_allr{}_b{}_d{}".format(
        exp_name or "offpolicy_ac",
        env_name,
        hidden_sizes,
        actor_learning_rate,
        critic_learning_rate,
        alpha_learning_rate,
        batch_size,
        discount,
    )

    log_string += f"_tem{temperature}"
    log_string += f"_stu{ema_decay}"  # soft_target_update_rate

    num_train_steps = int(
        (total_steps - training_starts) * num_train_steps_per_env_step
    )

    return {
        "agent_kwargs": {
            "hidden_sizes": hidden_sizes,
            "actor_learning_rate": actor_learning_rate,
            "critic_learning_rate": critic_learning_rate,
            "alpha_learning_rate": alpha_learning_rate,
            "num_train_steps": num_train_steps,
            "discount": discount,
            "ema_decay": ema_decay,
            "num_critics": num_critics,
            "num_critic_updates": num_critic_updates,
            "temperature": temperature,
            "num_actor_samples": num_actor_samples,
            "saved_path": saved_path,
            "seed": seed,
        },
        "replay_buffer_capacity": replay_buffer_capacity,
        "log_name": log_string,
        "total_steps": total_steps,
        "random_steps": random_steps,
        "training_starts": training_starts,
        "ep_len": ep_len,
        "batch_size": batch_size,
        "make_env": make_env,
        "num_train_steps_per_env_step": num_train_steps_per_env_step,
        "seed": seed,
    }
