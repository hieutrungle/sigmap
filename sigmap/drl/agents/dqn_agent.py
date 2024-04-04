from typing import Sequence, Callable, Tuple, Optional
import torch
from torch import nn
import numpy as np
import sigmap.drl.infrastructure.pytorch_utils as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Return the action to take in the given observation, using an epsilon-greedy
        policy.

        Parameters
        ----------
        observation: np.ndarray
            The observation to act on.
        epsilon: float
            The probability of taking a random action.

        Returns
        -------
        int
            The action to take.
        """
        observation = ptu.from_numpy(np.asanyarray(observation))[None]

        # if np.random.random() < epsilon:
        #     action = np.random.randint(self.num_actions)
        # else:
        #     action = self.critic(observation).argmax(dim=1).item()

        # return action
        if np.random.random() < epsilon:
            action = torch.tensor(np.random.choice(self.num_actions))
        else:
            action = self.critic(observation)
            action = action.argmax()

        return ptu.to_numpy(action).squeeze(0).item()

    def update_critic(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminateds: np.ndarray,
        truncateds: np.ndarray,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = rewards.shape
        dones = terminateds | truncateds
        # with torch.no_grad():
        #     next_qa_values = self.target_critic(next_observations)
        #     if self.use_double_q:
        #         # Use the critic instead of a 2nd target critic to select
        #         # the best action for the next state
        #         next_actions = self.critic(next_observations).argmax(dim=1)
        #     else:
        #         next_actions = next_qa_values.argmax(dim=1)

        #     next_q_values = torch.gather(
        #         next_qa_values, 1, next_actions[:, None]
        #     ).squeeze(1)
        #     target_q_values = rewards + self.discount * (1 - dones) * next_q_values

        with torch.no_grad():
            next_qa_values = self.target_critic(next_observations)

            if self.use_double_q:
                next_actions = self.critic(next_observations).argmax(dim=1)
            else:
                next_actions = next_qa_values.argmax(dim=1)

            next_q_values = next_qa_values.gather(1, next_actions[:, None]).squeeze(1)
            target_q_values = rewards + self.discount * next_q_values * (1 - dones)

        qa_values = self.critic(observations)
        q_values = torch.gather(qa_values, 1, actions[:, None]).squeeze(1)

        critic_loss = self.critic_loss(q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()
        self.lr_scheduler.step()

        return {
            "critic_loss": critic_loss.item(),
            "q_values": q_values.mean().item(),
            "target_q_values": target_q_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminateds: np.ndarray,
        truncateds: np.ndarray,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        critic_stats = self.update_critic(
            observations, actions, rewards, next_observations, terminateds, truncateds
        )

        if step % self.target_update_period == 0:
            self.update_target_critic()

        return critic_stats
