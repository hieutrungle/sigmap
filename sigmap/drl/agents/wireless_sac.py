from typing import Callable, Optional, Tuple, Sequence
import copy
import torch
import torch.nn as nn
import numpy as np
from sigmap.drl.infrastructure import pytorch_utils as ptu
from typing import Union
from sigmap.drl.infrastructure.distributions import (
    make_tanh_transformed,
    make_multi_normal,
    make_scaled_tanh_transformed,
)
from sigmap.drl.infrastructure.data_types import Observations

Activation = Union[str, nn.Module]

_str_to_activation = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "softplus": nn.Softplus(),
    "identity": nn.Identity(),
}


class Actor(nn.Module):
    def __init__(
        self,
        observation_shapes: Tuple[Sequence[int]],
        action_shape: Sequence[int],
        n_layers: int = 5,
        size: int = 128,
        activation: Activation = "tanh",
        output_activation: Activation = "identity",
        scale: float = 1.0,
        # state_dependent_std: bool = False,
        # fixed_std: Optional[float] = None,
    ):
        super().__init__()

        self.scale = scale

        if isinstance(activation, str):
            activation = _str_to_activation[activation]
        if isinstance(output_activation, str):
            output_activation = _str_to_activation[output_activation]

        # self.state_dependent_std = state_dependent_std
        # self.fixed_std = fixed_std

        self.focal_pts_shape = observation_shapes[0]
        self.tx_position_shape = observation_shapes[1]
        self.rx_position_shape = observation_shapes[2]

        n_lower_layers = n_layers // 3
        self.action_shape = action_shape

        # focal points
        in_size = np.prod(self.focal_pts_shape)
        upper_layers = []
        n_upper_layers = n_layers - n_lower_layers
        for _ in range(n_upper_layers):
            upper_layers.append(nn.Linear(in_size, size))
            upper_layers.append(activation)
            in_size = size
        self.upper_block = nn.Sequential(*upper_layers)

        # tx_poxsition + rx_position
        in_size = np.prod(self.tx_position_shape) + np.prod(self.rx_position_shape)
        position_layers = []
        for _ in range(1):
            position_layers.append(nn.Linear(in_size, size))
            position_layers.append(activation)
            in_size = size
        self.position_block = nn.Sequential(*position_layers)

        # Combine the two blocks
        in_size = size + size
        lower_layers = []
        for _ in range(n_lower_layers):
            lower_layers.append(nn.Linear(in_size, size))
            lower_layers.append(activation)
            in_size = size
        lower_layers.append(nn.Linear(in_size, np.prod(self.action_shape) * 2))
        lower_layers.append(output_activation)
        self.lower_block = nn.Sequential(*lower_layers)

        self.focal_pts_flatten_len = np.prod(self.focal_pts_shape)
        self.tx_position_flatten_len = np.prod(self.tx_position_shape)
        self.rx_position_flatten_len = np.prod(self.rx_position_shape)

        # if self.state_dependent_std:
        #     lower_layers.append(nn.Linear(in_size, np.prod(action_shape) * 2))
        # else:
        #     lower_layers.append(nn.Linear(in_size, np.prod(action_shape)))

        # if self.fixed_std:
        #     self.std = 0.1
        # else:
        #     self.std = nn.Parameter(
        #         torch.full(
        #             (np.prod(action_shape),),
        #             0.0,
        #             dtype=torch.float32,
        #             device=ptu.DEVICE,
        #         )
        #     )

    def forward(self, observations: dict) -> torch.distributions.Distribution:

        focal_pts: torch.Tensor = observations["focal_pts"]
        tx_positions: torch.Tensor = observations["tx_positions"]
        rx_positions: torch.Tensor = observations["rx_positions"]

        # Flatten the inputs
        focal_pts = focal_pts.view(
            *focal_pts.shape[: -len(self.focal_pts_shape)], self.focal_pts_flatten_len
        )
        tx_positions = tx_positions.view(
            *tx_positions.shape[: -len(self.tx_position_shape)],
            self.tx_position_flatten_len,
        )
        rx_positions = rx_positions.view(
            *rx_positions.shape[: -len(self.rx_position_shape)],
            self.rx_position_flatten_len,
        )

        focal = self.upper_block(focal_pts)
        positions = self.position_block(torch.cat([tx_positions, rx_positions], dim=-1))
        # Combine the two blocks
        # if self.state_dependent_std: means shape: (batch_size, 2 * ac_dim)
        # else: means shape: (batch_size, ac_dim)
        combined = torch.cat([focal, positions], dim=-1)
        mean_std = self.lower_block(combined)

        mean, std = torch.chunk(mean_std, 2, dim=-1)
        std: torch.Tensor = torch.nn.functional.softplus(std) + 1e-2

        # Convert to the correct shape
        mean = mean.view((*mean.shape[:-1], *self.action_shape))
        std = std.view((*std.shape[:-1], *self.action_shape))

        # if self.state_dependent_std:
        #     mean, std = torch.chunk(means, 2, dim=-1)
        #     std = torch.nn.functional.softplus(std) + 1e-2
        # else:
        #     mean = means
        #     if self.fixed_std:
        #         std = self.std
        #     else:
        #         std = torch.nn.functional.softplus(self.std) + 1e-2

        action_distribution = make_scaled_tanh_transformed(
            mean, std, self.scale, len(self.action_shape)
        )

        return action_distribution


class Critic(nn.Module):
    def __init__(
        self,
        observation_shapes: Tuple[Sequence[int]],
        action_shape: Sequence[int],
        n_layers: int = 5,
        size: int = 128,
        activation: Activation = "tanh",
        output_activation: Activation = "identity",
    ):
        super().__init__()
        if isinstance(activation, str):
            activation = _str_to_activation[activation]
        if isinstance(output_activation, str):
            output_activation = _str_to_activation[output_activation]

        # Shape without batch dimension
        self.focal_pts_shape = observation_shapes[0]
        self.tx_position_shape = observation_shapes[1]
        self.rx_position_shape = observation_shapes[2]
        self.action_shape = action_shape

        n_lower_layers = n_layers // 3

        # focal points
        in_size = np.prod(self.focal_pts_shape) + np.prod(self.action_shape)
        upper_layers = []
        n_upper_layers = n_layers - n_lower_layers
        for _ in range(n_upper_layers):
            upper_layers.append(nn.Linear(in_size, size))
            upper_layers.append(activation)
            in_size = size
        self.upper_block = nn.Sequential(*upper_layers)

        # tx_poxsition + rx_position
        in_size = np.prod(self.tx_position_shape) + np.prod(self.rx_position_shape)
        position_layers = []
        for _ in range(1):
            position_layers.append(nn.Linear(in_size, size))
            position_layers.append(activation)
            in_size = size
        self.position_block = nn.Sequential(*position_layers)

        # Combine the two blocks
        in_size = size + size
        lower_layers = []
        for _ in range(n_lower_layers):
            lower_layers.append(nn.Linear(in_size, size))
            lower_layers.append(activation)
            in_size = size
        lower_layers.append(nn.Linear(in_size, 1))
        lower_layers.append(output_activation)
        self.lower_block = nn.Sequential(*lower_layers)

        self.focal_pts_flatten_len = np.prod(self.focal_pts_shape)
        self.tx_position_flatten_len = np.prod(self.tx_position_shape)
        self.rx_position_flatten_len = np.prod(self.rx_position_shape)
        self.action_flatten_len = np.prod(self.action_shape)

    def forward(
        self,
        observations: dict,
        actions: torch.Tensor,
    ):
        focal_pts: torch.Tensor = observations["focal_pts"]
        tx_positions: torch.Tensor = observations["tx_positions"]
        rx_positions: torch.Tensor = observations["rx_positions"]

        batch_size = focal_pts.shape[0]

        # Flatten the inputs
        focal_pts = focal_pts.view(
            *focal_pts.shape[: -len(self.focal_pts_shape)], self.focal_pts_flatten_len
        )
        tx_positions = tx_positions.view(
            *tx_positions.shape[: -len(self.tx_position_shape)],
            self.tx_position_flatten_len,
        )
        rx_positions = rx_positions.view(
            *rx_positions.shape[: -len(self.rx_position_shape)],
            self.rx_position_flatten_len,
        )
        actions = actions.view(
            *actions.shape[: -len(self.action_shape)], self.action_flatten_len
        )

        mixed = torch.cat([focal_pts, actions], dim=-1)
        mixed = self.upper_block(mixed)
        positions = self.position_block(torch.cat([tx_positions, rx_positions], dim=-1))
        combined = torch.cat([mixed, positions], dim=-1)
        q_values = self.lower_block(combined)

        return q_values.squeeze(-1)


class SoftActorCritic(nn.Module):
    def __init__(
        self,
        observation_shapes: Tuple[Sequence[int]],
        action_shape: Sequence[int],
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_actor_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_critic_optimizer: Callable[
            [torch.nn.ParameterList], torch.optim.Optimizer
        ],
        make_critic_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: Optional[int] = None,
        soft_target_update_rate: Optional[float] = None,
        # Actor-critic configuration
        actor_gradient_type: str = "reinforce",  # One of "reinforce" or "reparametrize"
        num_actor_samples: int = 1,
        num_critic_updates: int = 1,
        # Settings for multiple critics
        num_critic_networks: int = 1,
        target_critic_backup_type: str = "mean",  # One of "doubleq", "min", "redq", or "mean"
        # Soft actor-critic
        use_entropy_bonus: bool = False,
        temperature: float = 0.0,
        backup_entropy: bool = True,
        action_scale: float = 1.0,
    ):
        super().__init__()

        assert target_critic_backup_type in [
            "doubleq",
            "min",
            "mean",
            "redq",
        ], f"{target_critic_backup_type} is not a valid target critic backup type"

        assert actor_gradient_type in [
            "reinforce",
            "reparametrize",
        ], f"{actor_gradient_type} is not a valid type of actor gradient update"

        assert (
            target_update_period is not None or soft_target_update_rate is not None
        ), "Must specify either target_update_period or soft_target_update_rate"

        # Actor
        # Actor receives a dict of {focal_pts, tx_position, rx_position}
        # and outputs a distribution of "delta_focal_pts"
        # "delta_focal_pts" shape: (batch_size, num_devices, 2, 3)
        self.action_scale = action_scale
        self.actor = Actor(observation_shapes, action_shape, scale=action_scale).to(
            ptu.DEVICE
        )
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.actor_lr_scheduler = make_actor_schedule(self.actor_optimizer)

        # Multiple Critics
        self.critics = nn.ModuleList(
            [
                Critic(observation_shapes, action_shape).to(ptu.DEVICE)
                for _ in range(num_critic_networks)
            ]
        )
        self.critics_optimizer = make_critic_optimizer(self.critics.parameters())
        self.critics_lr_scheduler = make_critic_schedule(self.critics_optimizer)
        self.target_critics = nn.ModuleList(
            [
                Critic(observation_shapes, action_shape).to(ptu.DEVICE)
                for _ in range(num_critic_networks)
            ]
        )

        self.observation_shapes = observation_shapes
        self.action_shape = action_shape
        self.discount = discount
        self.target_update_period = target_update_period
        self.target_critic_backup_type = target_critic_backup_type
        self.num_critic_networks = num_critic_networks
        self.use_entropy_bonus = use_entropy_bonus
        self.temperature = temperature
        self.actor_gradient_type = actor_gradient_type
        self.num_actor_samples = num_actor_samples
        self.num_critic_updates = num_critic_updates
        self.soft_target_update_rate = soft_target_update_rate
        self.backup_entropy = backup_entropy

        self.critic_loss = nn.MSELoss()

        self.update_target_critics()

        # Moving average reward
        self.moving_average_reward = None

    def get_action(self, observation: dict[np.ndarray]) -> np.ndarray:
        """
        Compute an action for a given observation.
        """
        observation = ptu.add_batch_dimension(observation)
        observation = ptu.from_numpy(observation)
        with torch.no_grad():
            action_distribution: torch.distributions.Distribution = self.actor(
                observation
            )
            action: torch.Tensor = action_distribution.sample()

        assert action.shape == (1, *self.action_shape), action.shape
        action: np.ndarray = ptu.to_numpy(action)
        return action.squeeze(0)

    def run_critics(
        self,
        observations: dict[torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the (ensembled) Q-values for the given state-action pair.

        q_values shape: (num_critics, batch_size)
        """
        q_values = [critic(observations, actions) for critic in self.critics]
        return torch.stack(q_values, dim=0)

    def run_target_critics(
        self,
        observations: dict[torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the (ensembled) target Q-values for the given state-action pair.

        q_values shape: (num_critics, batch_size)
        """
        q_values = [critic(observations, actions) for critic in self.target_critics]
        return torch.stack(q_values, dim=0)

    def q_backup_strategy(self, next_qs: torch.Tensor) -> torch.Tensor:
        """
        Handle Q-values from multiple different target critic networks to produce target values.
        For example:
         - for "vanilla", we can just leave the Q-values as-is (we only have one critic).
         - for double-Q, swap the critics' predictions (so each uses the other as the target).
         - for clip-Q, clip to the minimum of the two critics' predictions.
        Parameters:
            next_qs (torch.Tensor): Q-values of shape (num_critics, batch_size).
                Leading dimension corresponds to target values FROM the different critics.
        Returns:
            torch.Tensor: Target values of shape (num_critics, batch_size).
                Leading dimension corresponds to target values FOR the different critics.
        """

        assert (
            next_qs.ndim == 2
        ), f"next_qs should have shape (num_critics, batch_size) but got {next_qs.shape}"
        num_critic_networks, batch_size = next_qs.shape
        assert num_critic_networks == self.num_critic_networks

        if self.target_critic_backup_type == "doubleq":
            # Dual Q-update trick
            # Swap target_Q1 and target_Q2
            assert self.num_critic_networks == 2
            next_qs = torch.stack([next_qs[1], next_qs[0]], dim=0)

        elif self.target_critic_backup_type == "min":
            # Clipped Q-update
            assert self.num_critic_networks == 2
            next_qs, _ = torch.min(next_qs, dim=0)

        elif self.target_critic_backup_type == "mean":
            # Mean Q-update
            next_qs = torch.mean(next_qs, dim=0)

        elif self.target_critic_backup_type == "redq":
            # Subsample update
            num_min_qs = 2
            subsampled_next_qs = torch.gather(
                next_qs,
                dim=0,
                index=torch.randint(
                    low=0,
                    high=self.num_critic_networks,
                    size=(num_min_qs, batch_size),
                    device=ptu.device,
                ),
            )
            next_qs, _ = torch.min(subsampled_next_qs, dim=0)
        else:
            # No backup strategy, keep the Q-values as-is
            pass

        # If our backup strategy removed a dimension, add it back in explicitly
        # (assume the target for each critic will be the same)
        if next_qs.shape == (batch_size,):
            next_qs = (
                next_qs[None]
                .expand((self.num_critic_networks, batch_size))
                .contiguous()
            )

        assert next_qs.shape == (
            self.num_critic_networks,
            batch_size,
        ), next_qs.shape
        return next_qs

    def update_critics(
        self,
        observations: dict[torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: dict[torch.Tensor],
        dones: torch.Tensor,
    ):
        """
        Update the critic networks by computing target values and minimizing Bellman error.
        """
        (batch_size,) = rewards.shape
        # Compute target values
        # Important: we don't need gradients for target values!
        with torch.no_grad():
            next_actions_distribution: torch.distribution.Distribution = self.actor(
                next_observations
            )
            next_actions = next_actions_distribution.sample()
            next_qs = self.run_target_critics(next_observations, next_actions)
            next_qs = self.q_backup_strategy(next_qs)

            assert next_qs.shape == (
                self.num_critic_networks,
                batch_size,
            ), next_qs.shape

            if self.use_entropy_bonus and self.backup_entropy:
                # Add entropy bonus to the target values for SAC
                # Make sure to use the temperature parameter!
                # Hint: Make sure your entropy bonus has compatible dimensions! (Watch out for broadcasting)
                next_actions_entropy = self.entropy(next_actions_distribution)

                next_actions_entropy = (
                    next_actions_entropy[None]
                    .expand((self.num_critic_networks, batch_size))
                    .contiguous()
                )
                assert (
                    next_actions_entropy.shape == next_qs.shape
                ), next_actions_entropy.shape
                next_qs += self.temperature * next_actions_entropy

            # Compute target Q-values
            # rewards = (
            #     rewards[None]
            #     .expand((self.num_critic_networks, batch_size))
            #     .contiguous()
            # )
            # dones = (
            #     dones[None].expand((self.num_critic_networks, batch_size)).contiguous()
            # )
            # rewards = self.dB2linear(rewards)
            # next_qs = self.dB2linear(next_qs)
            # target_values: torch.Tensor = rewards + self.discount * next_qs * (
            #     1 - 1.0 * dones
            # )
            # target_values = self.linear2dB(target_values)  # in dB

            advantages = rewards - self.moving_average_reward
            advantages = (
                advantages[None]
                .expand((self.num_critic_networks, batch_size))
                .contiguous()
            )
            dones = (
                dones[None].expand((self.num_critic_networks, batch_size)).contiguous()
            )
            advantages = self.dB2linear(advantages)
            next_qs = self.dB2linear(next_qs)
            target_values: torch.Tensor = advantages + self.discount * next_qs * (
                1 - 1.0 * dones
            )
            target_values = self.linear2dB(target_values)  # in dB

            assert target_values.shape == (
                self.num_critic_networks,
                batch_size,
            ), target_values.shape

        # Update critics
        q_values = self.run_critics(observations, actions)
        assert q_values.shape == (self.num_critic_networks, batch_size), q_values.shape

        critics_loss: torch.Tensor = self.critic_loss(q_values, target_values)
        self.critics_optimizer.zero_grad()
        critics_loss.backward()
        self.critics_optimizer.step()

        return {
            "critics_loss": critics_loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
        }

    def entropy(self, action_distribution: torch.distributions.Distribution):
        """
        Compute the (approximate) entropy of the action distribution for each batch element.
        """
        # rsample() is a stochastic version of sample() that uses reparameterization trick

        samples = action_distribution.rsample((self.num_actor_samples,))
        log_probs = action_distribution.log_prob(samples)
        entropy_est = -torch.mean(log_probs, dim=0)
        return entropy_est

    def actor_loss_reinforce(self, observations: dict[torch.Tensor]) -> torch.Tensor:
        """
        Compute the REINFORCE loss for the actor.
        """
        batch_size = observations.shape[0]
        # Compute the action distribution
        action_distribution: torch.distributions.Distribution = self.actor(observations)

        with torch.no_grad():
            # Sample an action
            actions = action_distribution.rsample(
                sample_shape=(self.num_actor_samples,)
            )
            assert actions.shape == (
                self.num_actor_samples,
                batch_size,
                self.action_dim,
            ), actions.shape
            q_values = self.run_critics(
                observations[None].repeat((self.num_actor_samples, 1, 1)), actions
            )
            assert q_values.shape == (
                self.num_critic_networks,
                self.num_actor_samples,
                batch_size,
            ), q_values.shape

            # Our best guess of the Q-values is the mean of the ensemble
            # shape: (num_actor_samples, batch_size)
            q_values = torch.mean(q_values, axis=0)
            advantage = q_values

        log_probs = action_distribution.log_prob(actions)
        loss = -(log_probs * advantage).mean()

        return loss, torch.mean(self.entropy(action_distribution))

    def actor_loss_reparametrize(
        self, observations: dict[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the reparametrize loss for the actor.
        """
        batch_size = len(observations)
        action_distribution: torch.distributions.Distribution = self.actor(observations)

        # Sample actions
        actions = action_distribution.rsample(sample_shape=(self.num_actor_samples,))

        observations = self._replicate_observations(
            observations, self.num_actor_samples
        )
        q_values = self.run_critics(observations, actions).view(-1)

        # observations = observations[None].expand((self.num_actor_samples, -1, -1))
        # q_values = self.run_critics(observations, actions).view(-1)

        loss = torch.mean(-q_values)
        return loss, torch.mean(self.entropy(action_distribution))
        # return 0.0, 0.0

    def _replicate_observations(
        self, observations: dict[torch.Tensor], num_replicas: int
    ):
        """
        Replicate the observations to match the number of replicas.
        """
        observations = {
            k: v[None].expand((num_replicas, *v.shape)) for k, v in observations.items()
        }
        return observations

    def update_actor(self, obs: dict[torch.Tensor]):
        """
        Update the actor by one gradient step using either REPARAMETRIZE or REINFORCE.
        """

        if self.actor_gradient_type == "reparametrize":
            loss, entropy = self.actor_loss_reparametrize(obs)
        elif self.actor_gradient_type == "reinforce":
            loss, entropy = self.actor_loss_reinforce(obs)

        # Add entropy if necessary
        if self.use_entropy_bonus:
            loss = loss - self.temperature * entropy

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {"actor_loss": loss.item(), "entropy": entropy.item()}

    def update_target_critics(self):
        """
        Update target critics with current critics.
        """
        self.soft_update_target_critics(1.0)

    def soft_update_target_critics(self, tau):
        """
        Update target critics with moving average of current critics.
        """
        for target_critic, critic in zip(self.target_critics, self.critics):
            for target_param, param in zip(
                target_critic.parameters(), critic.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def update(
        self,
        observations: dict[torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: dict[torch.Tensor],
        dones: torch.Tensor,
        step: int,
    ):
        """
        Update the actor and critic networks.
        """

        # Moving average reward
        self.moving_average_reward = -100
        # if self.moving_average_reward is None:
        #     self.moving_average_reward = rewards.mean().item()
        # else:
        #     self.moving_average_reward = (
        #         self.moving_average_reward * 0.99 + rewards.mean().item() * 0.01
        #     )

        critic_infos = []
        for _ in range(self.num_critic_updates):
            info = self.update_critics(
                observations, actions, rewards, next_observations, dones
            )
            critic_infos.append(info)

        actor_info = self.update_actor(observations)

        if (
            self.target_update_period is not None
            and step % self.target_update_period == 0
        ):
            # Hard update target critics
            self.update_target_critics()
        elif self.soft_target_update_rate is not None:
            # Soft update target critics
            self.soft_update_target_critics(self.soft_target_update_rate)

        # Average the critic info over all of the steps
        critic_info = {
            k: np.mean([info[k] for info in critic_infos]) for k in critic_infos[0]
        }

        # Deal with LR scheduling
        self.actor_lr_scheduler.step()
        self.critics_lr_scheduler.step()

        return {
            **actor_info,
            **critic_info,
            "actor_lr": self.actor_lr_scheduler.get_last_lr()[0],
            "critics_lr": self.critics_lr_scheduler.get_last_lr()[0],
            "moving_average_reward": self.moving_average_reward,
        }

    def save(self, path: str, step: int):
        """
        Save the actor and critic networks to a file.
        """
        torch.save(
            {
                "step": step,
                "actor": self.actor.state_dict(),
                "critics": [critic.state_dict() for critic in self.critics],
            },
            path,
        )

    def linear2dB(self, x: torch.Tensor) -> torch.Tensor:
        return 10 * torch.log10(x)

    def dB2linear(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pow(10, torch.div(x, 10))
