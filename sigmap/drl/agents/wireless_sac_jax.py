from typing import Callable, Optional, Tuple, Sequence, Union, Any
import os
from jax._src.typing import Array
from jax._src import dtypes

KeyArray = Array
DTypeLikeFloat = Any
DTypeLikeComplex = Any
DTypeLikeInexact = Any  # DTypeLikeFloat | DTypeLikeComplex

import numpy as np

import jax
from jax import lax, random, numpy as jnp
from flax import struct
from jax._src import core
from flax import linen as nn  # nn notation also used in PyTorch and in Flax's older API

from flax.training import train_state
import orbax.checkpoint as ocp

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
    hidden_sizes: Sequence[int] = (128, 128, 128)
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
        skip_focals = nn.Dense(self.hidden_sizes[-2])(focals)
        for hidden_size in self.hidden_sizes[:-1]:
            focals = nn.Dense(hidden_size)(focals)
            focals = _str_to_activation[self.activation](focals)
        focals = skip_focals + focals

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
        # dist = D.Normal(means, jnp.exp(log_stds))
        # dist = D.Transformed(dist, D.Tanh())
        # dist = D.Independent(dist, reinterpreted_batch_ndims=len(self.action_shape))
        # return dist
        return means, log_stds


class Critic(nn.Module):
    """Critic network for SAC."""

    observation_shapes: dict[str, Sequence[int]]
    action_shape: Sequence[int]
    hidden_sizes: Sequence[int] = (128, 128, 128)
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
        skip_mixed = nn.Dense(self.hidden_sizes[-2])(mixed)
        for hidden_size in self.hidden_sizes[:-1]:
            mixed = nn.Dense(hidden_size)(mixed)
            mixed = _str_to_activation[self.activation](mixed)
        mixed = skip_mixed + mixed

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


def alpha_init(
    key: KeyArray,
    shape: core.Shape,
    temperature: float,
    dtype: DTypeLikeInexact = jnp.float_,
) -> Array:
    """An initializer that returns a constant array full of ones.

    The ``key`` argument is ignored.

    >>> import jax, jax.numpy as jnp
    >>> jax.nn.initializers.ones(jax.random.key(42), (3, 2), jnp.float32)
    Array([[1., 1.],
           [1., 1.],
           [1., 1.]], dtype=float32)
    """
    return jnp.ones(shape, dtypes.canonicalize_dtype(dtype)) * temperature


class Alpha(nn.Module):
    """
    Temperature parameter for entropy.
    """

    temperature: float = 0.05
    alpha_init: Callable = alpha_init

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        alpha = self.param(
            "alpha",  # parametar name (as it will appear in the FrozenDict)
            self.alpha_init,  # initialization function, RNG passed implicitly through init fn
            (x.shape[-1], 1),
            self.temperature,
        )  # shape info

        x = jnp.dot(x, alpha)
        x = jnp.mean(x)
        x = jnp.clip(x, 0.0001, 0.1)
        return x


@jax.tree_util.register_pytree_node_class
@dataclass
class SoftActorCritic:
    """
    Soft Actor-Critic agent.
    """

    actor_state: train_state.TrainState
    critic_states: list[train_state.TrainState]
    target_critic_states: list[train_state.TrainState]
    alpha_state: train_state.TrainState
    key: jax.random.PRNGKey
    discount: float = 0.99
    # tau: float = 0.005
    ema_decay: float = 0.995
    num_critics: int = 2
    num_critic_updates: int = 5
    target_entropy: float = -10.0
    num_actor_samples: int = 5
    checkpoint_manager: ocp.CheckpointManager = None

    @classmethod
    def create(
        cls,
        observation_shapes: dict[str, Sequence[int]],
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (128, 128, 128),
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        alpha_learning_rate: float = 1e-4,
        num_train_steps: int = None,
        discount: float = 0.99,
        # tau: float = 0.005,  # soft target update rate
        ema_decay: float = 0.995,  # soft target update rate
        num_critics: int = 2,
        num_critic_updates: int = 5,
        temperature: float = 0.05,  # temperature for entropy
        num_actor_samples: int = 5,
        saved_path: Optional[str] = None,
        seed: int = 0,
    ):
        key = random.PRNGKey(seed)
        tmp_observations = {}
        for k, shape in observation_shapes.items():
            tmp_observations[k] = jax.random.normal(key, (1, *shape))
        tmp_actions = jax.random.normal(key, (1, *action_shape))

        @optax.inject_hyperparams
        def chain_optimizer(learning_rate: float):
            return optax.chain(
                optax.clip(1.0), optax.adamw(learning_rate=learning_rate)
            )

        def create_optimizer(
            learning_rate: float, num_train_steps: int = None
        ) -> optax.GradientTransformation:
            init_value = learning_rate / 50
            end_value = learning_rate / 10
            if num_train_steps == None:
                num_train_steps = 1_000
            warmup_steps = num_train_steps // 5
            decay_steps = num_train_steps
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=init_value,
                peak_value=learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_value=end_value,
            )
            optimizer = chain_optimizer(learning_rate=schedule)
            return optimizer

        # Alpha
        target_entropy = -np.prod(action_shape)
        alpha = Alpha(temperature=temperature)
        alpha_opt = create_optimizer(alpha_learning_rate, num_train_steps)
        alpha_state = train_state.TrainState.create(
            apply_fn=alpha.apply,
            params=alpha.init(key, jnp.array(1.0).reshape(1, 1)),
            tx=alpha_opt,
        )

        # Actor
        actor = Actor(
            observation_shapes=observation_shapes,
            action_shape=action_shape,
            hidden_sizes=hidden_sizes,
        )
        actor_opt = create_optimizer(actor_learning_rate, num_train_steps)
        actor_state = train_state.TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(key, tmp_observations),
            tx=actor_opt,
        )

        # Critics
        critics = [
            Critic(
                observation_shapes=observation_shapes,
                action_shape=action_shape,
                hidden_sizes=hidden_sizes,
            )
            for _ in range(num_critics)
        ]
        critic_opt = create_optimizer(
            critic_learning_rate, int(num_train_steps * num_critic_updates)
        )
        critic_states = []
        target_critic_states = []
        for critic in critics:
            key, subkey = random.split(key)
            params = critic.init(subkey, tmp_observations, tmp_actions)
            critic_state = train_state.TrainState.create(
                apply_fn=critic.apply,
                params=params,
                tx=critic_opt,
            )
            target_critic_state = train_state.TrainState(
                step=0,
                apply_fn=critic.apply,
                params=params,
                tx=None,
                opt_state=None,
            )
            critic_states.append(critic_state)
            target_critic_states.append(target_critic_state)

        # checkpoint orbax
        if saved_path is None:
            saved_path = "./tmp/sac"
            # get absolute path
        saved_path = os.path.abspath(saved_path)
        options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
        orbax_checkpointer = ocp.StandardCheckpointer()
        checkpoint_manager = ocp.CheckpointManager(
            saved_path,
            # orbax_checkpointer,
            options=options,
        )

        return cls(
            actor_state,
            critic_states,
            target_critic_states,
            alpha_state,
            key,
            discount,
            ema_decay,
            num_critics,
            num_critic_updates,
            target_entropy,
            num_actor_samples,
            checkpoint_manager,
        )

    def make_action_distribution(
        self, means: jnp.ndarray, log_stds: jnp.ndarray, reinterpreted_batch_ndims=3
    ) -> D.Distribution:
        action_dist = D.Normal(means, jnp.exp(log_stds))
        action_dist = D.Transformed(action_dist, D.Tanh())
        action_dist = D.Independent(
            action_dist, reinterpreted_batch_ndims=reinterpreted_batch_ndims
        )
        return action_dist

    def get_action_distribution(
        self,
        observations: dict[np.ndarray],
        actor_params: struct.PyTreeNode,
        actor_apply_fn: Callable,
    ) -> D.Distribution:
        """
        Compute an action distribution for a given observation.
        """
        means, log_stds = actor_apply_fn(actor_params, observations)
        return self.make_action_distribution(
            means, log_stds, reinterpreted_batch_ndims=3
        )

    @jax.jit
    def _get_action(
        self,
        observation: dict[np.ndarray],
        actor_state: train_state.TrainState,
        key: jax.random.PRNGKey,
    ) -> np.ndarray:
        rep_observation = self._replicate_observations(observation, num_repeats=1)
        action_dist = self.get_action_distribution(
            rep_observation, actor_state.params, actor_state.apply_fn
        )
        action = action_dist.sample(seed=key)
        return jnp.squeeze(action, axis=0)

    def get_action(self, observation: dict[np.ndarray]) -> np.ndarray:
        """
        Compute an action for a given observation.
        """
        action = self._get_action(observation, self.actor_state, self.key)
        return np.array(action)

    def update_target_critics(self):
        """
        Update target critics with current critics.
        """
        return self.soft_update_target_critics(
            1.0, self.target_critic_states, self.critic_states, self.num_critics
        )

    @functools.partial(jax.jit, static_argnums=(4))
    def soft_update_target_critics(
        self,
        ema_decay: float,
        target_critic_states: list[train_state.TrainState],
        critic_states: list[train_state.TrainState],
        num_critics: int,
    ):
        """
        Update target critics with moving average of current critics.
        """
        for i in range(num_critics):
            old = target_critic_states[i].params
            new = critic_states[i].params
            new_target_params = jax.tree.map(
                lambda x, y: (1 - ema_decay) * x + y * ema_decay, new, old
            )
            target_critic_states[i] = target_critic_states[i].replace(
                step=target_critic_states[i].step + 1, params=new_target_params
            )
        return target_critic_states

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
        next_action_distribution: D.Distribution = self.get_action_distribution(
            next_observations, actor_state.params, actor_state.apply_fn
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
        alpha = alpha_state.apply_fn(alpha_state.params, jnp.array(1.0).reshape(1, 1))

        next_q_values = next_q_values + alpha * next_action_entropy

        # lower_bound = -100.0  # dB
        lower_bound = 0
        advantages = rewards - lower_bound

        # Expand rewards and dones to match the number of critics
        advantages = self._expand_repeat(advantages, self.num_critics)
        dones = self._expand_repeat(dones, self.num_critics)

        # Loss
        advantages = self.dB2linear(advantages)
        next_q_values = self.dB2linear(next_q_values)
        target_q_values = advantages + self.discount * (1 - dones) * next_q_values
        target_q_values = self.linear2dB(target_q_values)

        q_values = self.run_critics(
            list_critic_params, list_critic_apply_fn, observations, actions
        )

        loss = jnp.mean((q_values - target_q_values) ** 2)

        return loss, (
            target_q_values,
            q_values,
            next_action_entropy,
            self.linear2dB(next_q_values),
        )

    def _expand_repeat(self, x, num_repeats):
        x = jnp.expand_dims(x, axis=0)
        x = jnp.repeat(x, num_repeats, axis=0)
        return x

    def calc_entropy(
        self,
        action_distribution: D.Distribution,
        key: jax.random.PRNGKey,
    ):
        """
        Compute the (approximate) entropy of the action distribution for each batch element.
        """
        _, log_probs = action_distribution.sample_and_log_prob(
            seed=key, sample_shape=(self.num_actor_samples,)
        )

        entropy_est = -jnp.mean(log_probs, axis=0)
        return entropy_est

    @jax.jit
    def update_critics(
        self,
        observations: dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: dict[str, np.ndarray],
        dones: np.ndarray,
        critic_states: list[train_state.TrainState],
        target_critic_states: list[train_state.TrainState],
        actor_state: train_state.TrainState,
        alpha_state: train_state.TrainState,
        key: jax.random.PRNGKey,
    ):
        """
        Update the critic networks by computing target values and minimizing Bellman error.
        """
        critic_apply_fns = tuple([critic.apply_fn for critic in critic_states])
        target_critic_apply_fns = tuple(
            [critic.apply_fn for critic in target_critic_states]
        )
        loss_fn = lambda list_critic_params: self.calc_critic_loss(
            list_critic_params,
            critic_apply_fns,
            [critic.params for critic in target_critic_states],
            target_critic_apply_fns,
            actor_state,
            alpha_state,
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            key,
        )
        (loss, (target_q_values, q_values, entropy, next_q_values)), grads = (
            jax.value_and_grad(loss_fn, has_aux=True)(
                [critic.params for critic in critic_states]
            )
        )

        for i, critic_state in enumerate(critic_states):
            critic_states[i] = critic_state.apply_gradients(grads=grads[i])
        return critic_states, {
            "critic loss": loss,
            "target_q_values": jnp.mean(target_q_values),
            "q_values": jnp.mean(q_values),
            "next_q_values": jnp.mean(next_q_values),
            "critic entropy": jnp.mean(entropy),
        }

    def calc_actor_loss(
        self,
        actor_params: struct.PyTreeNode,
        actor_apply_fn: Callable,
        critic_states: train_state.TrainState,
        alpha_state: train_state.TrainState,
        observations: dict[str, np.ndarray],
        key: jax.random.PRNGKey,
    ):
        action_distribution: D.Distribution = self.get_action_distribution(
            observations, actor_params, actor_apply_fn
        )
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
        alpha = alpha_state.apply_fn(alpha_state.params, jnp.array(1.0).reshape(1, 1))

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

    @jax.jit
    def update_actor(
        self,
        observations: dict[str, np.ndarray],
        actor_state: train_state.TrainState,
        critic_states: list[train_state.TrainState],
        alpha_state: train_state.TrainState,
        key: jax.random.PRNGKey,
    ):
        loss_fn = lambda params: self.calc_actor_loss(
            params,
            actor_state.apply_fn,
            critic_states,
            alpha_state,
            observations,
            key,
        )
        (loss, (entropy, alpha)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            actor_state.params
        )
        actor_state = actor_state.apply_gradients(grads=grads)
        return actor_state, {
            "actor loss": loss,
            "actor entropy": entropy,
            "alpha": alpha,
        }

    def calc_alpha_loss(
        self,
        alpha_params: struct.PyTreeNode,
        alpha_apply_fn: Callable,
        actor_state: train_state.TrainState,
        target_entropy: np.ndarray,
        observations: dict[str, np.ndarray],
        key: jax.random.PRNGKey,
    ):
        """
        Update the temperature parameter alpha.
        """
        action_distribution: D.Distribution = self.get_action_distribution(
            observations, actor_state.params, actor_state.apply_fn
        )

        # Loss
        alpha = alpha_apply_fn(alpha_params, jnp.array(1.0).reshape(1, 1))

        entropy = self.calc_entropy(action_distribution, key)
        entropy = jnp.mean(entropy)
        loss = jnp.mean(alpha * (entropy - target_entropy))

        return loss, alpha

    @jax.jit
    def update_alpha(
        self,
        observations: dict[str, np.ndarray],
        alpha_state: train_state.TrainState,
        actor_state: train_state.TrainState,
        target_entropy: float,
        key: jax.random.PRNGKey,
    ):
        """
        Update the temperature parameter alpha.
        """

        loss_fn = lambda params: self.calc_alpha_loss(
            params,
            alpha_state.apply_fn,
            actor_state,
            target_entropy,
            observations,
            key,
        )

        (loss, alpha), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            alpha_state.params
        )

        alpha_state = alpha_state.apply_gradients(grads=grads)
        return alpha_state, {
            "alpha loss": loss,
            "new_alpha": alpha_state.params["params"]["alpha"].mean(),
        }

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
            self.critic_states, critic_info = self.update_critics(
                observations,
                actions,
                rewards,
                next_observations,
                dones,
                self.critic_states,
                self.target_critic_states,
                self.actor_state,
                self.alpha_state,
                self.key,
            )
            self.target_critic_states = self.soft_update_target_critics(
                self.ema_decay,
                self.target_critic_states,
                self.critic_states,
                self.num_critics,
            )

            critic_infos.append(critic_info)

        self.actor_state, actor_info = self.update_actor(
            observations,
            self.actor_state,
            self.critic_states,
            self.alpha_state,
            self.key,
        )

        self.alpha_state, alpha_info = self.update_alpha(
            observations,
            self.alpha_state,
            self.actor_state,
            self.target_entropy,
            self.key,
        )

        critic_info = {
            k: np.mean([info[k] for info in critic_infos]) for k in critic_infos[0]
        }

        actor_lr = self.actor_state.opt_state.hyperparams["learning_rate"]
        critic_lr = self.critic_states[0].opt_state.hyperparams["learning_rate"]
        alpha_lr = self.alpha_state.opt_state.hyperparams["learning_rate"]

        return {
            **actor_info,
            **critic_info,
            **alpha_info,
            "actor_lr": actor_lr,
            "critics_lr": critic_lr,
            "alpha_lr": alpha_lr,
        }

    def save(self, step: int):
        """
        Save the agent's parameters to a file.
        """
        self.checkpoint_manager.save(step, args=ocp.args.StandardSave(self))

    def wait_for_checkpoint(self):
        """
        Wait for the checkpoint manager to finish writing checkpoints.
        """
        self.checkpoint_manager.wait_until_finished()

    def load(self, step: int = None):
        """
        Load the agent's parameters from a file.
        """
        if step == None:
            step = self.checkpoint_manager.best_step()
        return self.checkpoint_manager.restore(
            step, args=ocp.args.StandardRestore(self)
        )

    def tree_flatten(self):
        # first group (if it's non-hashable/dynamic)
        # or the second group (if it's hashable/static)

        # arrays / dynamic values
        # children = (self.actor_state, self.critic_states, self.target_critic_states)
        children = (
            self.actor_state,
            self.critic_states,
            self.target_critic_states,
            self.alpha_state,
        )

        # static values
        aux_data = (
            self.key,
            self.discount,
            self.ema_decay,
            self.num_critics,
            self.num_critic_updates,
            self.target_entropy,
            self.num_actor_samples,
            self.checkpoint_manager,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

    def linear2dB(self, x):
        return 10 * jnp.log10(x)

    def dB2linear(self, x):
        return jnp.power(10, x / 10)
