from typing import Callable, Optional, Tuple, Sequence
import copy
import torch
import torch.nn as nn
import numpy as np
from sigmap.drl.infrastructure import pytorch_utils as ptu


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

        self.update_target_critic()

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute the action for a given observation.
        """
        with torch.no_grad():
            # add batch dimension
            observation = ptu.from_numpy(observation)[None]

            action_distribution: torch.distributions.Distribution = self.actor(
                observation
            )
            action: torch.Tensor = action_distribution.sample()

            assert action.shape == (1, self.action_dim), action.shape
            return ptu.to_numpy(action).squeeze(0)

    def run_critics(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the (ensembled) Q-values for the given state-action pair.
        """
        q_values = [[critic(observations, actions)] for critic in self.critics]
        return torch.stack(q_values, dim=0)

    def run_target_critics(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the (ensembled) target Q-values for the given state-action pair.
        """
        q_values = [[critic(observations, actions)] for critic in self.target_critics]
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
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        done: torch.Tensor,
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

            # Compute target Q-values
            target_values: torch.Tensor = rewards[None] + self.discount * next_qs * (
                1 - 1.0 * done[None]
            )
            assert target_values.shape == (
                self.num_critic_networks,
                batch_size,
            ), target_values.shape

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
                next_qs -= self.temperature * next_actions_entropy

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
        return -action_distribution.log_prob(action_distribution.rsample())

    def actor_loss_reinforce(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Compute the REINFORCE loss for the actor.
        """
        batch_size = observations.shape[0]
        # Compute the action distribution
        action_distribution: torch.distributions.Distribution = self.actor(observations)

        with torch.no_grad():
            # Sample an action
            actions = action_distribution.sample(sample_shape=(self.num_actor_samples,))
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

        # Do REINFORCE: calculate log-probs and use the Q-values
        log_probs: torch.Tensor = action_distribution.log_prob(actions)
        torch.nan_to_num_(log_probs, nan=0.0, posinf=0.0, neginf=0.0)
        assert log_probs.shape == (
            self.num_actor_samples,
            batch_size,
        ), log_probs.shape

        loss = torch.mean(-(advantage * log_probs))

        return loss, torch.mean(self.entropy(action_distribution))

    def actor_loss_reparametrize(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Compute the reparametrize loss for the actor.
        """
        batch_size = observations.shape[0]
        # Compute the action distribution
        action_distribution: torch.distributions.Distribution = self.actor(observations)

        actions = action_distribution.rsample(sample_shape=(self.num_actor_samples,))
        assert actions.shape == (
            self.num_actor_samples,
            batch_size,
            self.action_dim,
        ), actions.shape

        q_values = self.critic(
            observations[None].repeat((self.num_actor_samples, 1, 1)), actions
        )
        assert q_values.shape == (
            self.num_critic_networks,
            self.num_actor_samples,
            batch_size,
        ), q_values.shape

        loss = torch.mean(-q_values)
        return loss, torch.mean(self.entropy(action_distribution))

    def update_actor(self, obs: torch.Tensor):
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
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
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
