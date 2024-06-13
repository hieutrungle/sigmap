from typing import Callable, Optional, Tuple, Sequence, Union
import numpy as np

import jax
from jax import lax, random, numpy as jnp
from flax import core, struct
from flax.core import freeze, unfreeze
from flax import linen as nn  # nn notation also used in PyTorch and in Flax's older API
from flax.training import train_state  # a useful dataclass to keep train state

# JAX optimizers - a separate lib developed by DeepMind
import optax

import functools
from dataclasses import dataclass

from sigmap.drl import distributions as D

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
        dist = D.Normal(means, jnp.exp(log_stds))
        dist = D.Transformed(dist, D.Tanh())
        dist = D.Independent(dist, reinterpreted_batch_ndims=len(self.action_shape))
        return dist


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


class Alpha(nn.Module):
    """
    Temperature parameter for entropy.
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(1)(x)
        x = jnp.mean(x)
        x = jnp.clip(x, 0.0001, 0.1)
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
    temperature: float = 0.05  # temperature for entropy
    num_actor_samples: int = 5
    seed: int = 0

    def __init__(
        self,
        observation_shapes: dict[str, Sequence[int]],
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int],
        actor_learning_rate: float,
        critic_learning_rate: float,
        alpha_learning_rate: float,
        discount: float,
        tau: float,
        num_critics: int = 2,
        num_critic_updates: int = 5,
        target_critic_backup_type: int = 1,
        temperature: float = 0.05,  # temperature for entropy
        num_actor_samples: int = 5,
        seed: int = 0,
    ):
        super().__init__()
        self.observation_shapes = observation_shapes
        self.action_shape = action_shape
        self.hidden_sizes = hidden_sizes
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.alpha_learning_rate = alpha_learning_rate
        self.discount = discount
        self.tau = tau
        self.num_critics = num_critics
        self.num_critic_updates = num_critic_updates
        self.target_critic_backup_type = target_critic_backup_type
        self.temperature = temperature
        self.num_actor_samples = num_actor_samples
        self.seed = seed

        self.key = random.PRNGKey(self.seed)
        tmp_observations = {}
        for key, shape in self.observation_shapes.items():
            tmp_observations[key] = jax.random.normal(self.key, (1, *shape))
        tmp_actions = jax.random.normal(self.key, (1, *self.action_shape))

        # Alpha
        self.target_entropy = -np.prod(action_shape)
        alpha = Alpha()
        alpha_opt = optax.adamw(self.alpha_learning_rate)
        self.alpha_state = train_state.TrainState.create(
            apply_fn=alpha.apply,
            params=alpha.init(self.key, jnp.array(self.temperature).reshape(1, 1)),
            tx=alpha_opt,
        )

        # Actor
        actor = Actor(
            observation_shapes=self.observation_shapes,
            action_shape=self.action_shape,
            hidden_sizes=self.hidden_sizes,
        )
        actor_opt = optax.adamw(self.actor_learning_rate)
        self.actor_state = train_state.TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(self.key, tmp_observations),
            tx=actor_opt,
        )

        # Critics
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

    @functools.partial(jax.jit, static_argnums=(2))
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

    @functools.partial(jax.jit, static_argnums=(2, 4))
    def calc_critic_loss(
        self,
        list_critic_params: list[struct.PyTreeNode],
        list_critic_apply_fn: Tuple[Callable],
        list_target_critic_params: list[struct.PyTreeNode],
        list_target_critic_apply_fn: Tuple[Callable],
        actor_state: train_state.TrainState,
        alpha_state: train_state.TrainState,
        observations: dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: dict[str, np.ndarray],
        dones: np.ndarray,
        key: jax.random.PRNGKey,
    ):

        # Target Q-values
        next_action_distribution: D.Distribution = actor_state.apply_fn(
            actor_state.params, next_observations
        )
        next_actions = next_action_distribution.sample(seed=key)
        next_q_values = self.run_critics(
            list_target_critic_params,
            list_target_critic_apply_fn,
            next_observations,
            next_actions,
        )
        next_q_values = self.do_q_backup(next_q_values)

        # Entropy
        next_action_entropy = self.calc_entropy(next_action_distribution, key)
        next_action_entropy = self._expand_repeat(next_action_entropy, self.num_critics)
        alpha = alpha_state.apply_fn(
            alpha_state.params, jnp.array(self.temperature).reshape(1, 1)
        )

        next_q_values = next_q_values + alpha * next_action_entropy

        # Expand rewards and dones to match the number of critics
        rewards = self._expand_repeat(rewards, self.num_critics)
        dones = self._expand_repeat(dones, self.num_critics)

        # Loss
        target_q_values = rewards + self.discount * (1 - dones) * next_q_values

        q_values = self.run_critics(
            list_critic_params, list_critic_apply_fn, observations, actions
        )

        loss = jnp.mean((q_values - target_q_values) ** 2)

        return loss, next_action_entropy

    def _expand_repeat(self, x, num_repeats):
        x = jnp.expand_dims(x, axis=0)
        x = jnp.repeat(x, num_repeats, axis=0)
        return x

    @functools.partial(jax.jit, static_argnums=(2, 4))
    def calc_critic_loss_grad(
        self,
        list_critic_params: list[struct.PyTreeNode],
        list_critic_apply_fn: Tuple[Callable],
        list_target_critic_params: list[struct.PyTreeNode],
        list_target_critic_apply_fn: Tuple[Callable],
        actor_state: train_state.TrainState,
        alpha_state: train_state.TrainState,
        observations: dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: dict[str, np.ndarray],
        dones: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        grad_fn = jax.value_and_grad(self.calc_critic_loss, has_aux=True)
        return grad_fn(
            list_critic_params,
            list_critic_apply_fn,
            list_target_critic_params,
            list_target_critic_apply_fn,
            actor_state,
            alpha_state,
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            key,
        )

    @jax.jit
    def apply_grad_critics(self, grads, critic_states):

        for i, grad in enumerate(grads):
            grads[i] = jax.tree_map(lambda x: jnp.clip(x, -1.0, 1.0), grad)

        for i, critic_state in enumerate(critic_states):
            critic_states[i] = critic_state.apply_gradients(grads=grads[i])

        return critic_states

    @jax.jit
    def calc_entropy(
        self,
        action_distribution: D.Distribution,
        key: jax.random.PRNGKey,
    ):
        """
        Compute the (approximate) entropy of the action distribution for each batch element.
        """
        samples = action_distribution.sample(
            seed=key, sample_shape=(self.num_actor_samples,)
        )
        log_probs = action_distribution.log_prob(samples)

        entropy_est = -jnp.mean(log_probs, axis=0)
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
        (loss, entropy), grads = self.calc_critic_loss_grad(
            [critic.params for critic in self.critic_states],
            critic_apply_fns,
            [critic.params for critic in self.target_critic_states],
            target_critic_apply_fns,
            self.actor_state,
            self.alpha_state,
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            self.key,
        )
        self.critic_states = self.apply_grad_critics(grads, self.critic_states)
        return {"critic loss": loss, "critic entropy": entropy}

    @functools.partial(jax.jit, static_argnums=(2))
    def calc_actor_loss(
        self,
        actor_params: struct.PyTreeNode,
        actor_apply_fn: Callable,
        critic_states: train_state.TrainState,
        alpha_state: train_state.TrainState,
        observations: dict[str, np.ndarray],
        key: jax.random.PRNGKey,
    ):
        # Q-values
        action_distribution: D.Distribution = actor_apply_fn(actor_params, observations)
        actions = action_distribution.sample(
            seed=key, sample_shape=(self.num_actor_samples,)
        )

        rep_observations = self._replicate_observations(
            observations, self.num_actor_samples
        )

        # shape: (num_critics, num_actor_samples, batch_size)
        q_values = self.run_critics(
            [critic_state.params for critic_state in critic_states],
            tuple([critic_state.apply_fn for critic_state in critic_states]),
            rep_observations,
            actions,
        )

        # Alpha
        alpha = alpha_state.apply_fn(
            alpha_state.params, jnp.array(self.temperature).reshape(1, 1)
        )

        # Loss
        entropy = self.calc_entropy(action_distribution, key)
        entropy = jnp.mean(entropy)
        loss = -jnp.mean(q_values) - alpha * entropy

        return loss, (entropy, alpha)

    def _replicate_observations(
        self,
        observations: dict[str, np.ndarray],
        num_repeats: int,
    ):
        """
        Replicate the observations to match the number of replicas.
        """
        rep_observations = {}
        for key, value in observations.items():
            rep_observations[key] = self._expand_repeat(value, num_repeats)
        return rep_observations

    @functools.partial(jax.jit, static_argnums=(2))
    def calc_actor_loss_grad(
        self,
        actor_params: struct.PyTreeNode,
        actor_apply_fn: Callable,
        critic_states: train_state.TrainState,
        alpha_state: train_state.TrainState,
        observations: dict[str, np.ndarray],
        key: jax.random.PRNGKey,
    ):
        grad_fn = jax.value_and_grad(self.calc_actor_loss, has_aux=True)
        return grad_fn(
            actor_params, actor_apply_fn, critic_states, alpha_state, observations, key
        )

    def update_actor(
        self,
        observations: dict[str, np.ndarray],
    ):
        (loss, (entropy, alpha)), grads = self.calc_actor_loss_grad(
            self.actor_state.params,
            self.actor_state.apply_fn,
            self.critic_states,
            self.alpha_state,
            observations,
            self.key,
        )
        self.actor_state = self.actor_state.apply_gradients(grads=grads)
        return {
            "actor loss": loss,
            "actor entropy": entropy,
            "alpha": alpha,
        }

    def calc_alpha_loss(
        self,
        alpha_params: struct.PyTreeNode,
        alpha_apply_fn: Callable,
        actor_state: train_state.TrainState,
        temperature: np.ndarray,
        target_entropy: np.ndarray,
        observations: dict[str, np.ndarray],
        key: jax.random.PRNGKey,
    ):
        """
        Update the temperature parameter alpha.
        """
        # Q-values
        action_distribution: D.Distribution = actor_state.apply_fn(
            actor_state.params, observations
        )

        # Loss
        alpha = alpha_apply_fn(alpha_params, jnp.array(temperature).reshape(1, 1))

        entropy = self.calc_entropy(action_distribution, key)
        entropy = jnp.mean(entropy)
        loss = jnp.mean(alpha * (entropy - target_entropy))

        return loss, alpha

    @functools.partial(jax.jit, static_argnums=(2))
    def calc_alpha_loss_grad(
        self,
        alpha_params: struct.PyTreeNode,
        alpha_apply_fn: Callable,
        actor_state: train_state.TrainState,
        temperature: np.ndarray,
        target_entropy: np.ndarray,
        observations: dict[str, np.ndarray],
        key: jax.random.PRNGKey,
    ):
        grad_fn = jax.value_and_grad(self.calc_alpha_loss, has_aux=True)
        return grad_fn(
            alpha_params,
            alpha_apply_fn,
            actor_state,
            temperature,
            target_entropy,
            observations,
            key,
        )

    def update_alpha(self, observations: dict[str, np.ndarray]):
        """
        Update the temperature parameter alpha.
        """
        # Q-values
        (loss, alpha), grads = self.calc_alpha_loss_grad(
            self.alpha_state.params,
            self.alpha_state.apply_fn,
            self.actor_state,
            np.array(self.temperature),
            np.array(self.target_entropy),
            observations,
            self.key,
        )

        self.alpha_state = self.alpha_state.apply_gradients(grads=grads)
        return {"alpha loss": loss, "alpha": alpha}

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

        alpha_info = self.update_alpha(observations)

        critic_info = {
            k: np.mean([info[k] for info in critic_infos]) for k in critic_infos[0]
        }

        return {
            **actor_info,
            **critic_info,
            **alpha_info,
            # "actor_lr": self.actor_lr_scheduler.get_last_lr()[0],
            # "critics_lr": self.critics_lr_scheduler.get_last_lr()[0],
            # "moving_average_reward": self.moving_average_reward,
        }

    def tree_flatten(self):
        # first group (if it's non-hashable/dynamic)
        # or the second group (if it's hashable/static)

        # arrays / dynamic values
        # children = (self.actor_state, self.critic_states, self.target_critic_states)
        children = tuple([])

        # static values
        aux_data = (
            self.observation_shapes,
            self.action_shape,
            self.hidden_sizes,
            self.actor_learning_rate,
            self.critic_learning_rate,
            self.alpha_learning_rate,
            self.discount,
            self.tau,
            self.num_critics,
            self.num_critic_updates,
            self.target_critic_backup_type,
            self.temperature,
            self.num_actor_samples,
            self.seed,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)
