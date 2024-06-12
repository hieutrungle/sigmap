from typing import Callable, Optional, Tuple, Sequence, Union
import copy
import numpy as np
import timeit

import jax
from jax import lax, random, numpy as jnp
from flax import core, struct
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn  # nn notation also used in PyTorch and in Flax's older API
from flax.training import train_state  # a useful dataclass to keep train state
from jax import tree_util

# JAX optimizers - a separate lib developed by DeepMind
import optax

import functools
from jax.tree_util import register_pytree_node_class
from dataclasses import dataclass
import time
import time

Activation = Union[str, Callable]


class Identity(nn.Module):
    """Identity module for Flax."""

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


_str_to_activation = {
    "relu": nn.activation.relu,
    "tanh": nn.activation.tanh,
    "sigmoid": nn.activation.sigmoid,
    "swish": nn.activation.hard_swish,
    "gelu": nn.activation.gelu,
    "identity": Identity(),
}


class Fourier(nn.Module):
    """Fourier features for encoding the input signal."""

    num_features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        weights = self.param(
            "weights",
            nn.initializers.normal(),
            (x.shape[-1], self.num_features),
        )
        x = jnp.dot(2 * np.pi * x, weights)
        x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
        return x


class Actor(nn.Module):
    """Actor network for SAC."""

    observation_shapes: dict[str, Sequence[int]]
    action_shape: Sequence[int]
    hidden_sizes: Sequence[int]
    activation: Activation = "tanh"
    output_activation: Activation = "identity"

    @nn.compact
    def __call__(
        self, observations: dict[jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        focal_pts: jnp.ndarray = observations["focal_pts"]
        tx_positions: jnp.ndarray = observations["tx_positions"]
        rx_positions: jnp.ndarray = observations["rx_positions"]

        focal_pts_shape = focal_pts.shape
        tx_positions_shape = tx_positions.shape
        rx_positions_shape = rx_positions.shape

        focal_pts_flatten_len = np.prod(np.array(focal_pts_shape[-3:]))
        tx_positions_flatten_len = np.prod(np.array(tx_positions_shape[-2:]))
        rx_positions_flatten_len = np.prod(np.array(rx_positions_shape[-2:]))

        focal_pts = focal_pts.reshape((*focal_pts_shape[:-3], focal_pts_flatten_len))
        tx_positions = tx_positions.reshape(
            (*tx_positions_shape[:-2], tx_positions_flatten_len)
        )
        rx_positions = rx_positions.reshape(
            (*rx_positions_shape[:-2], rx_positions_flatten_len)
        )

        focals = Fourier(num_features=self.hidden_sizes[0] // 2)(focal_pts)
        for hidden_size in self.hidden_sizes[:-1]:
            focals = nn.Dense(hidden_size)(focals)
            focals = _str_to_activation[self.activation](focals)

        positions = jnp.concatenate([tx_positions, rx_positions], axis=-1)
        positions = Fourier(num_features=self.hidden_sizes[0] // 2)(positions)
        positions = nn.Dense(self.hidden_sizes[-2])(positions)
        positions = _str_to_activation[self.activation](positions)

        x = jnp.concatenate([focals, positions], axis=-1)
        x = nn.Dense(self.hidden_sizes[-1])(x)
        x = _str_to_activation[self.output_activation](x)

        action_flatten_len = np.prod(np.array(self.action_shape))
        means = nn.Dense(action_flatten_len)(x)
        log_stds = nn.Dense(action_flatten_len)(x)

        means = means.reshape((*means.shape[:-1], *self.action_shape))
        log_stds = log_stds.reshape((*log_stds.shape[:-1], *self.action_shape))
        return means, log_stds


class Critic(nn.Module):
    """Critic network for SAC."""

    observation_shapes: dict[str, Sequence[int]]
    action_shape: Sequence[int]
    hidden_sizes: Sequence[int]
    activation: Activation = "gelu"
    output_activation: Activation = "identity"

    @nn.compact
    def __call__(
        self, observations: dict[jnp.ndarray], actions: jnp.ndarray
    ) -> jnp.ndarray:
        focal_pts: jnp.ndarray = observations["focal_pts"]
        tx_positions: jnp.ndarray = observations["tx_positions"]
        rx_positions: jnp.ndarray = observations["rx_positions"]

        focal_pts_shape = focal_pts.shape
        tx_positions_shape = tx_positions.shape
        rx_positions_shape = rx_positions.shape

        focal_pts_flatten_len = np.prod(np.array(focal_pts_shape[-3:]))
        tx_positions_flatten_len = np.prod(np.array(tx_positions_shape[-2:]))
        rx_positions_flatten_len = np.prod(np.array(rx_positions_shape[-2:]))
        action_flatten_len = np.prod(np.array(self.action_shape))

        focal_pts = focal_pts.reshape((*focal_pts_shape[:-3], focal_pts_flatten_len))
        actions = actions.reshape((*actions.shape[:-3], action_flatten_len))
        tx_positions = tx_positions.reshape(
            (*tx_positions_shape[:-2], tx_positions_flatten_len)
        )
        rx_positions = rx_positions.reshape(
            (*rx_positions_shape[:-2], rx_positions_flatten_len)
        )

        mixed = jnp.concatenate([focal_pts, actions], axis=-1)
        mixed = Fourier(num_features=self.hidden_sizes[0] // 2)(mixed)
        for hidden_size in self.hidden_sizes[:-1]:
            mixed = nn.Dense(hidden_size)(mixed)
            mixed = _str_to_activation[self.activation](mixed)

        positions = jnp.concatenate([tx_positions, rx_positions], axis=-1)
        positions = Fourier(num_features=self.hidden_sizes[0] // 2)(positions)
        positions = nn.Dense(self.hidden_sizes[-2])(positions)
        positions = _str_to_activation[self.activation](positions)

        x = jnp.concatenate([mixed, positions], axis=-1)
        x = nn.Dense(self.hidden_sizes[-1])(x)
        x = _str_to_activation[self.activation](x)
        x = nn.Dense(1)(x)
        x = _str_to_activation[self.output_activation](x)

        return x


@jax.tree_util.register_pytree_node_class
@dataclass
class SoftActorCritic:
    """
    Soft Actor-Critic agent.
    """

    observation_shapes: dict[str, Sequence[int]]
    action_shape: Sequence[int]
    hidden_sizes: Sequence[int]
    actor_learning_rate: float
    critic_learning_rate: float
    discount: float
    tau: float
    num_critics: int = 2
    num_critic_updates: int = 5
    target_critic_backup_type: int = 1
    seed: int = 0

    def __init__(
        self,
        observation_shapes: dict[str, Sequence[int]],
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int],
        actor_learning_rate: float,
        critic_learning_rate: float,
        discount: float,
        tau: float,
        num_critics: int = 2,
        num_critic_updates: int = 5,
        target_critic_backup_type: int = 1,
        alpha: float = 0.05,  # temperature for entropy
        seed: int = 0,
    ):
        super().__init__()
        self.observation_shapes = observation_shapes
        self.action_shape = action_shape
        self.hidden_sizes = hidden_sizes
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.discount = discount
        self.tau = tau
        self.num_critics = num_critics
        self.num_critic_updates = num_critic_updates
        self.target_critic_backup_type = target_critic_backup_type
        self.alpha = alpha
        self.seed = seed

        self.key = random.PRNGKey(self.seed)
        tmp_observations = {}
        for key, shape in self.observation_shapes.items():
            tmp_observations[key] = jax.random.normal(self.key, (1, *shape))
        tmp_actions = jax.random.normal(self.key, (1, *self.action_shape))

        self.actor = Actor(
            observation_shapes=self.observation_shapes,
            action_shape=self.action_shape,
            hidden_sizes=self.hidden_sizes,
        )
        actor_opt = optax.adamw(self.actor_learning_rate)
        self.actor_state = train_state.TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(self.key, tmp_observations),
            tx=actor_opt,
        )

        critics = [
            Critic(
                observation_shapes=self.observation_shapes,
                action_shape=self.action_shape,
                hidden_sizes=self.hidden_sizes,
            )
            for _ in range(self.num_critics)
        ]
        critic_opt = optax.adamw(self.critic_learning_rate)
        self.critic_states = []
        self.target_critic_states = []
        for critic in critics:
            self.key, subkey = random.split(self.key)
            critic_state = train_state.TrainState.create(
                apply_fn=critic.apply,
                params=critic.init(subkey, tmp_observations, tmp_actions),
                tx=critic_opt,
            )
            target_critic_state = train_state.TrainState(
                step=0,
                apply_fn=critic.apply,
                params=critic.init(subkey, tmp_observations, tmp_actions),
                tx=None,
                opt_state=None,
            )
            self.critic_states.append(critic_state)
            self.target_critic_states.append(target_critic_state)

        self.update_target_critics()

    def update_target_critics(self):
        """
        Update target critics with current critics.
        """
        self.soft_update_target_critics(1.0)

    def soft_update_target_critics(self, tau):
        """
        Update target critics with moving average of current critics.
        """
        for i, critic_state in enumerate(self.critic_states):
            new_target_params = jax.tree.map(
                lambda x, y: (1 - tau) * x + y * tau,
                self.target_critic_states[i].params,
                critic_state.params,
            )
            self.target_critic_states[i] = self.target_critic_states[i].replace(
                step=self.target_critic_states[i].step + 1, params=new_target_params
            )

    def run_critics(
        self,
        list_critic_params: list[struct.PyTreeNode],
        list_critic_apply_fn: Tuple[Callable],
        observations: dict[str, np.ndarray],
        actions: np.ndarray,
    ):
        q_values = []
        for critic_params, critic_apply_fn in zip(
            list_critic_params, list_critic_apply_fn
        ):
            q_values.append(
                jnp.expand_dims(
                    jnp.squeeze(
                        critic_apply_fn(critic_params, observations, actions),
                        axis=-1,
                    ),
                    axis=0,
                )
            )
        q_values = jnp.concatenate(q_values, axis=0)
        return q_values

    @functools.partial(jax.jit, static_argnums=(2))
    def get_actions(
        self,
        actor_params: struct.PyTreeNode,
        actor_apply_fn: Callable,
        observations: dict[str, np.ndarray],
        key: jax.random.PRNGKey,
    ) -> np.ndarray:
        """
        Get actions from the agent's actor network.
        """
        means, log_stds = actor_apply_fn(actor_params, observations)
        actions = jax.random.normal(key, means.shape) * jnp.exp(log_stds) + means
        # take tanh transform to ensure action is in [-1, 1]
        actions = jnp.tanh(actions)
        return actions

    def calc_actor_loss(
        self, actor_params, actor_apply_fn, critic_state, observations, key
    ):
        actions = self.get_actions(actor_params, actor_apply_fn, observations, key)
        q_values = critic_state.apply_fn(critic_state.params, observations, actions)

        loss = -jnp.mean(q_values)
        return loss

    def do_q_backup(self, next_qs: jnp.ndarray):
        """
        Handle Q-values from multiple different target critic networks to produce target values.

        Clip-Q, clip to the minimum of the two critics' predictions.

        Parameters:
            next_qs (jnp.ndarray): Q-values of shape (num_critics, batch_size).
                Leading dimension corresponds to target values FROM the different critics.
        Returns:
            jnp.ndarray: Target values of shape (num_critics, batch_size).
                Leading dimension corresponds to target values FOR the different critics.
        """

        next_qs = jnp.min(next_qs, axis=0)
        next_qs = jnp.expand_dims(next_qs, axis=0)
        next_qs = jnp.repeat(next_qs, self.num_critics, axis=0)

        return next_qs

    # @functools.partial(jax.jit, static_argnums=(2, 4))
    def calc_critic_loss(
        self,
        list_critic_params: list[struct.PyTreeNode],
        list_critic_apply_fn: Tuple[Callable],
        list_target_critic_params: list[struct.PyTreeNode],
        list_target_critic_apply_fn: Tuple[Callable],
        actor_state: train_state.TrainState,
        observations: dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: dict[str, np.ndarray],
        dones: np.ndarray,
    ):
        next_actions = self.get_actions(
            actor_state.params,
            actor_state.apply_fn,
            next_observations,
            self.key,
        )
        next_q_values = self.run_critics(
            list_target_critic_params,
            list_target_critic_apply_fn,
            next_observations,
            next_actions,
        )
        next_q_values = self.do_q_backup(next_q_values)
        next_action_entropy = self.calc_entropy(
            actor_state.params, actor_state.apply_fn, next_observations, next_actions
        )
        next_action_entropy = self._expand_repeat(next_action_entropy, self.num_critics)
        next_q_values = next_q_values + self.alpha * next_action_entropy

        rewards = self._expand_repeat(rewards, self.num_critics)
        dones = self._expand_repeat(dones, self.num_critics)

        target_q_values = rewards + self.discount * (1 - dones) * next_q_values

        q_values = self.run_critics(
            list_critic_params, list_critic_apply_fn, observations, actions
        )

        loss = jnp.mean((q_values - target_q_values) ** 2)

        return loss

    def _expand_repeat(self, x, num_repeats):
        x = jnp.expand_dims(x, axis=0)
        x = jnp.repeat(x, num_repeats, axis=0)
        return x

    # @functools.partial(jax.jit, static_argnums=(2, 4))
    def calc_critic_loss_grad(
        self,
        list_critic_params: list[struct.PyTreeNode],
        list_critic_apply_fn: Tuple[Callable],
        list_target_critic_params: list[struct.PyTreeNode],
        list_target_critic_apply_fn: Tuple[Callable],
        observations,
        actions,
        rewards,
        next_observations,
        dones,
    ):
        grad_fn = jax.value_and_grad(self.calc_critic_loss)
        return grad_fn(
            list_critic_params,
            list_critic_apply_fn,
            list_target_critic_params,
            list_target_critic_apply_fn,
            self.actor_state,
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        )

    @jax.jit
    def apply_grad_critics(self, grads, critic_states):

        for i, grad in enumerate(grads):
            grads[i] = jax.tree_map(lambda x: jnp.clip(x, -1.0, 1.0), grad)

        for i, critic_state in enumerate(critic_states):
            critic_states[i] = critic_state.apply_gradients(grads=grads[i])

        return critic_states

    @functools.partial(jax.jit, static_argnums=(2))
    def calc_entropy(
        self,
        actor_params: struct.PyTreeNode,
        actor_apply_fn: Callable,
        observations: dict[str, np.ndarray],
        actions: np.ndarray,
    ):
        """
        Compute the (approximate) entropy of the action distribution for each batch element.
        """
        means, log_stds = actor_apply_fn(actor_params, observations)

        actions = jnp.arctanh(actions)

        # TODO: this is not correct, need to compute the log probability of the actions with batch shape and event shape
        log_probs = jax.scipy.stats.norm.logpdf(
            actions, loc=means, scale=jnp.exp(log_stds)
        )

        entropy_est = -jnp.mean(
            log_probs, axis=np.arange(1, len(self.action_shape) + 1)
        )
        return entropy_est

    def update_critics(
        self,
        observations: dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: dict[str, np.ndarray],
        dones: np.ndarray,
    ):
        """
        Update the critic networks by computing target values and minimizing Bellman error.
        """
        critic_apply_fns = tuple([critic.apply_fn for critic in self.critic_states])
        target_critic_apply_fns = tuple(
            [critic.apply_fn for critic in self.target_critic_states]
        )
        loss, grads = self.calc_critic_loss_grad(
            [critic.params for critic in self.critic_states],
            critic_apply_fns,
            [critic.params for critic in self.target_critic_states],
            target_critic_apply_fns,
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        )
        self.critic_states = self.apply_grad_critics(grads, self.critic_states)
        return {"loss": loss}

    def update_actor(
        self,
        observations: dict[str, np.ndarray],
    ):
        return {}

    def update(
        self,
        observations: dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: dict[str, np.ndarray],
        dones: np.ndarray,
        step: int,
    ) -> dict[str, float]:
        """
        Update the agent's actor and critic networks.
        """

        critic_infos = []
        for _ in range(self.num_critic_updates):
            info = self.update_critics(
                observations, actions, rewards, next_observations, dones
            )
            critic_infos.append(info)

        actor_info = self.update_actor(observations)

        self.soft_update_target_critics(self.tau)

        critic_info = {
            k: np.mean([info[k] for info in critic_infos]) for k in critic_infos[0]
        }

        return {
            **actor_info,
            **critic_info,
            # **alpha_info,
            # "actor_lr": self.actor_lr_scheduler.get_last_lr()[0],
            # "critics_lr": self.critics_lr_scheduler.get_last_lr()[0],
            # "moving_average_reward": self.moving_average_reward,
        }

    def tree_flatten(self):
        # first group (if it's non-hashable/dynamic)
        # or the second group (if it's hashable/static)
        # static values
        aux_data = (
            self.observation_shapes,
            self.action_shape,
            self.hidden_sizes,
            self.actor_learning_rate,
            self.critic_learning_rate,
            self.discount,
            self.tau,
            self.num_critics,
            self.num_critic_updates,
            self.target_critic_backup_type,
            self.alpha,
            self.seed,
        )
        # arrays / dynamic values
        # children = (self.actor_state, self.critic_states, self.target_critic_states)
        children = tuple([])
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)


# tree_util.register_pytree_node_class(
#     SoftActorCritic, SoftActorCritic.tree_flatten, SoftActorCritic.tree_unflatten
# )
