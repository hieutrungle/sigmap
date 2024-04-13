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
)

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
        observation_shapes: list[Tuple[int, ...]],
        action_shape: Tuple[int, ...],
        n_layers: int = 5,
        size: int = 128,
        activation: Activation = "tanh",
        output_activation: Activation = "identity",
        # state_dependent_std: bool = False,
        # fixed_std: Optional[float] = None,
    ):
        super().__init__()

        if isinstance(activation, str):
            activation = _str_to_activation[activation]
        if isinstance(output_activation, str):
            output_activation = _str_to_activation[output_activation]

        # self.state_dependent_std = state_dependent_std
        # self.fixed_std = fixed_std

        device_states_shape = observation_shapes[0]
        tx_position_shape = observation_shapes[1]
        rx_position_shape = observation_shapes[2]

        input_size = np.prod(device_states_shape)

        # Device states
        in_size = input_size
        upper_layers = []
        n_upper_layers = n_layers - (n_layers // 3)
        for _ in range(n_upper_layers):
            upper_layers.append(nn.Linear(in_size, size))
            upper_layers.append(activation)
            in_size = size
        self.upper_block = nn.Sequential(*upper_layers)

        # tx_poxsition + rx_position
        in_size = np.prod(tx_position_shape) + np.prod(rx_position_shape)
        position_layers = []
        for _ in range(min(n_layers // 3, 1)):
            position_layers.append(nn.Linear(in_size, size))
            position_layers.append(activation)
            in_size = size
        self.position_block = nn.Sequential(*position_layers)

        # Combine the two blocks
        in_size = size + size
        lower_layers = []
        for _ in range(n_layers // 3):
            lower_layers.append(nn.Linear(in_size, size))
            lower_layers.append(activation)
            in_size = size

        # if self.state_dependent_std:
        #     lower_layers.append(nn.Linear(in_size, np.prod(action_shape) * 2))
        # else:
        #     lower_layers.append(nn.Linear(in_size, np.prod(action_shape)))
        lower_layers.append(nn.Linear(in_size, np.prod(action_shape) * 2))
        lower_layers.append(output_activation)
        self.lower_block = nn.Sequential(*lower_layers)

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

    def forward(self, observations: list[dict[Union[np.ndarray, list]]]):

        # make batches
        device_states = []
        tx_positions = []
        rx_positions = []
        for obs in observations:
            device_states.append(ptu.from_numpy(np.array(obs["device_state"]))[None])
            tx_positions.append(ptu.from_numpy(np.array(obs["tx_position"]))[None])
            rx_positions.append(ptu.from_numpy(np.array(obs["rx_position"]))[None])
        device_states = torch.cat(device_states, dim=0)
        tx_positions = torch.cat(tx_positions, dim=0)
        rx_positions = torch.cat(rx_positions, dim=0)

        device_states_shape = device_states.shape
        tx_positions_shape = tx_positions.shape
        rx_positions_shape = rx_positions.shape

        # Flatten the inputs
        device_states = device_states.view(device_states_shape[0], -1)
        tx_positions = tx_positions.view(tx_positions_shape[0], -1)
        rx_positions = rx_positions.view(rx_positions_shape[0], -1)

        device_states = self.upper_block(device_states)
        positions = self.position_block(torch.cat([tx_positions, rx_positions], dim=1))
        # Combine the two blocks
        # if self.state_dependent_std: means shape: (batch_size, 2 * ac_dim)
        # else: means shape: (batch_size, ac_dim)
        mean_std = self.lower_block(torch.cat([device_states, positions], dim=1))

        mean, std = torch.chunk(mean_std, 2, dim=-1)
        std = torch.nn.functional.softplus(std) + 1e-2

        # Convert to the correct shape
        mean = mean.view(device_states_shape)
        std = std.view(device_states_shape)

        # if self.state_dependent_std:
        #     mean, std = torch.chunk(means, 2, dim=-1)
        #     std = torch.nn.functional.softplus(std) + 1e-2
        # else:
        #     mean = means
        #     if self.fixed_std:
        #         std = self.std
        #     else:
        #         std = torch.nn.functional.softplus(self.std) + 1e-2

        action_distribution = make_tanh_transformed(mean, std)

        return action_distribution


class Critic(nn.Module):
    def __init__(self):
        pass

    def forward(
        self,
        observations: list[dict[Union[np.ndarray, list]]],
        actions: list[np.ndarray],
    ):
        pass


class SoftActorCritic(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,
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
        # Actor receives a dict of {device_states, tx_position, rx_position}
        # and outputs a distribution of "delta_device_states"
        # "delta_device_states" shape: (batch_size, num_devices, num_tiles_per_device, controlled_elements)
        self.actor = make_actor(observation_shape, action_dim)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.actor_lr_scheduler = make_actor_schedule(self.actor_optimizer)

        # Multiple Critics
        self.critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )
        self.critics_optimizer = make_critic_optimizer(self.critics.parameters())
        self.critics_lr_scheduler = make_critic_schedule(self.critics_optimizer)
        self.target_critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )

        self.observation_shape = observation_shape
        self.action_dim = action_dim
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

    def get_action(
        self, observation: list[dict[Union[np.ndarray, list]]]
    ) -> np.ndarray:
        """
        Compute the action for a given observation.
        """
        # TODO: adapt to observation type list[dict[Union[np.ndarray, list]]]
        with torch.no_grad():

            action_distribution: torch.distributions.Distribution = self.actor(
                observation
            )
            action: torch.Tensor = action_distribution.sample()

            assert action.shape == (1, self.action_dim), action.shape
            return ptu.to_numpy(action).squeeze(0)

    def run_critics(
        self,
        observations: list[dict[Union[np.ndarray, list]]],
        actions: list[np.ndarray],
    ) -> torch.Tensor:
        """
        Compute the (ensembled) Q-values for the given state-action pair.
        """
        # TODO: adapt to observation type list[dict[Union[np.ndarray, list]]]
        q_values = [critic(observations, actions) for critic in self.critics]
        return torch.stack(q_values, dim=0)

    def run_target_critics(
        self,
        observations: list[dict[Union[np.ndarray, list]]],
        actions: list[np.ndarray],
    ) -> torch.Tensor:
        """
        Compute the (ensembled) target Q-values for the given state-action pair.
        """
        # TODO: adapt to observation type list[dict[Union[np.ndarray, list]]]
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
        observations: list[dict[Union[np.ndarray, list]]],
        actions: list[np.ndarray],
        rewards: list[np.ndarray],
        next_observations: list[dict[Union[np.ndarray, list]]],
        dones: list[np.ndarray],
    ):
        """
        Update the critic networks by computing target values and minimizing Bellman error.
        """
        # TODO: adapt to observation type list[dict[Union[np.ndarray, list]]]
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
            target_values: torch.Tensor = rewards[None] + self.discount * next_qs * (
                1 - 1.0 * dones[None]
            )
            assert target_values.shape == (
                self.num_critic_networks,
                batch_size,
            ), target_values.shape

        # Update critics
        q_values = self.run_critics(observations, actions)
        assert q_values.shape == (self.num_critic_networks, batch_size), q_values.shape

        critics_loss = self.critic_loss(q_values, target_values)
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

    def actor_loss_reinforce(
        self, observations: list[dict[Union[np.ndarray, list]]]
    ) -> torch.Tensor:
        """
        Compute the REINFORCE loss for the actor.
        """
        # TODO: adapt to observation type list[dict[Union[np.ndarray, list]]]
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
        self, observations: list[dict[Union[np.ndarray, list]]]
    ) -> torch.Tensor:
        """
        Compute the reparametrize loss for the actor.
        """
        # TODO: adapt to observation type list[dict[Union[np.ndarray, list]]]
        batch_size = len(observations)
        # action_distributions:
        # {
        #   "device_states": torch.distributions.Distribution,
        #   "tx_position": torch.distributions.Distribution,
        #   "rx_position": torch.distributions.Distribution
        # }
        action_distributions: dict[str, torch.distributions.Distribution] = self.actor(
            observations
        )
        delta_device_states_distribution = action_distributions["device_states"]
        delta_tx_position_distribution = action_distributions["tx_position"]
        delta_rx_position_distribution = action_distributions["rx_position"]

        # Sample actions
        delta_device_states = delta_device_states_distribution.rsample(
            sample_shape=(self.num_actor_samples,)
        )
        delta_tx_position = delta_tx_position_distribution.rsample(
            sample_shape=(self.num_actor_samples,)
        )
        delta_rx_position = delta_rx_position_distribution.rsample(
            sample_shape=(self.num_actor_samples,)
        )

        device_states_shape = np.array(observations[0]["device_states"]).shape
        tx_position_shape = np.array(observations[0]["tx_position"]).shape
        rx_position_shape = np.array(observations[0]["rx_position"]).shape

        assert delta_device_states.shape == (
            self.num_actor_samples,
            batch_size,
            *device_states_shape,
        ), delta_device_states.shape
        assert delta_tx_position.shape == (
            self.num_actor_samples,
            batch_size,
            *tx_position_shape,
        ), delta_tx_position.shape
        assert delta_rx_position.shape == (
            self.num_actor_samples,
            batch_size,
            *rx_position_shape,
        ), delta_rx_position.shape

        actions = {
            "delta_device_states": delta_device_states,
            "delta_tx_position": delta_tx_position,
            "delta_rx_position": delta_rx_position,
        }
        # TODO: Implement the rest of the function
        observations = self._replicate_observations(
            observations, self.num_actor_samples
        )
        q_values = self.run_critics(observations, actions).view(-1)

        # observations = observations[None].expand((self.num_actor_samples, -1, -1))
        # q_values = self.run_critics(observations, actions).view(-1)

        loss = torch.mean(-q_values)
        return loss, torch.mean(self.entropy(action_distributions))
        # return 0.0, 0.0

    def _replicate_observations(
        self, observations: list[dict[Union[np.ndarray, list]]], num_replicas: int
    ):
        """
        Replicate the observations to match the number of replicas.
        """
        return [copy.deepcopy(observations) for _ in range(num_replicas)]

    def update_actor(self, obs: list[dict[Union[np.ndarray, list]]]):
        """
        Update the actor by one gradient step using either REPARAMETRIZE or REINFORCE.
        """

        if self.actor_gradient_type == "reparametrize":
            loss, entropy = self.actor_loss_reparametrize(obs)
        elif self.actor_gradient_type == "reinforce":
            loss, entropy = self.actor_loss_reinforce(obs)

        # Add entropy if necessary
        if self.use_entropy_bonus:
            loss -= self.temperature * entropy

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
        observations: list[dict[Union[np.ndarray, list]]],
        actions: list[np.ndarray],
        rewards: list[np.ndarray],
        next_observations: list[dict[Union[np.ndarray, list]]],
        dones: list[np.ndarray],
        step: int,
    ):
        """
        Update the actor and critic networks.
        """

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
        }
