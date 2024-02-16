from typing import Callable, Optional, Tuple
import numpy as np
import torch.nn as nn
import torch
from sigmap.drl.infrastructure import pytorch_utils as ptu


class ModelBasedAgent(nn.Module):
    def __init__(
        self,
        make_dynamics_model: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[nn.ParameterList], torch.optim.Optimizer],
        ensemble_size: int,
        mpc_horizon: int,
        mpc_strategy: str,
        mpc_num_action_sequences: int,
        cem_num_iters: Optional[int] = None,
        cem_num_elites: Optional[int] = None,
        cem_alpha: Optional[float] = None,
    ):
        super().__init__()
        self.mpc_horizon = mpc_horizon
        self.mpc_strategy = mpc_strategy
        self.mpc_num_action_sequences = mpc_num_action_sequences
        self.cem_num_iters = cem_num_iters
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        assert mpc_strategy in (
            "cem",
            "random",
        ), f"Invalid MPC strategy: {mpc_strategy}"

        self.ob_dim = 1
        self.ac_dim = 1
        self.ensemble_size = ensemble_size
        self.dynamics_models = nn.ModuleList(
            [
                make_dynamics_model(self.ob_dim, self.ob_dim)
                for _ in range(ensemble_size)
            ]
        )
        self.optimizer = make_optimizer(self.dynamics_models.parameters())
        self.loss_fn = nn.MSELoss()

        # keep track of statistics for both the model input (obs & act) and
        # output (obs delta)
        self.register_buffer(
            "obs_acs_mean", torch.zeros(self.ob_dim + self.ac_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_acs_std", torch.ones(self.ob_dim + self.ac_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_delta_mean", torch.zeros(self.ob_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_delta_std", torch.ones(self.ob_dim, device=ptu.device)
        )

    def update(
        self, i: int, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray
    ) -> np.ndarray:
        """
        Update self.dynamics_models[i] using the given batch of data.

        Args:
            i: index of the dynamics model to update
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
            next_obs: (batch_size, ob_dim)

        Returns:
            loss: the loss of the dynamics model on the given batch of data
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        next_obs = ptu.from_numpy(next_obs)

        # train the dynamics model with normalized inputs and outputs
        eps = 1e-8
        obs_acs = torch.cat([obs, acs], dim=1)
        obs_acs_normalized = (obs_acs - self.obs_acs_mean) / (self.obs_acs_std + eps)
        pred_deltas_normalized = self.dynamics_models[i](obs_acs_normalized)

        deltas = next_obs - obs
        deltas_normalized = (deltas - self.obs_delta_mean) / (self.obs_delta_std + eps)

        loss = self.loss_fn(pred_deltas_normalized, deltas_normalized)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return ptu.to_numpy(loss)

    @torch.no_grad()
    def update_statistics(self, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray):
        """
        Update the statistics used to normalize the inputs and outputs of the dynamics models.

        Args:
            obs: (n, ob_dim)
            acs: (n, ac_dim)
            next_obs: (n, ob_dim)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        next_obs = ptu.from_numpy(next_obs)
        self.obs_acs_mean = torch.mean(torch.cat([obs, acs], dim=1), dim=0)
        self.obs_acs_std = torch.std(torch.cat([obs, acs], dim=1), dim=0)
        self.obs_delta_mean = torch.mean(next_obs - obs, dim=0)
        self.obs_delta_std = torch.std(next_obs - obs, dim=0)

    @torch.no_grad()
    def get_dynamics_predictions(
        self, i: int, obs: np.ndarray, acs: np.ndarray
    ) -> np.ndarray:
        """
        Takes a batch of observations and actions and returns the predicted
        next observations from the ith dynamics model - self.dynamics_models[i].

        Args:
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
        Returns: (batch_size, ob_dim)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)

        eps = 1e-8
        obs_acs = torch.cat([obs, acs], dim=1)
        obs_acs_normalized = (obs_acs - self.obs_acs_mean) / (self.obs_acs_std + eps)
        pred_deltas_normalized = self.dynamics_models[i](obs_acs_normalized)
        pred_deltas = (
            pred_deltas_normalized * (self.obs_delta_std + eps) + self.obs_delta_mean
        )
        pred_next_obs = obs + pred_deltas
        return ptu.to_numpy(pred_next_obs)

    def get_reward(self, acs: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
        """
        Compute the reward for each action sequence using the reward function.

        Args:
            acs: (ensemble_size, mpc_num_action_sequences, ac_dim)
            next_obs: (ensemble_size, mpc_num_action_sequences, ob_dim)
        Returns:
            rewards: (mpc_num_action_sequences,)
        """
        raise NotImplementedError

    def evaluate_action_sequences(self, obs: np.ndarray, action_sequences: np.ndarray):
        """
        Evaluate the action sequences using the ensemble of learned dynamics models.

        mpc_num_action_sequences: number of action sequences to evaluate
        mpc_num_action_sequences = batch_size

        Args:
            obs: (ob_dim,)
            action_sequence: (mpc_num_action_sequences, mpc_horizon, ac_dim)
        Returns:
            sum_of_rewards: (mpc_num_action_sequences,)
        """
        # We are going to predict (ensemble_size * mpc_num_action_sequences)
        # distinct rollouts, and then average over the ensemble dimension to get
        # the reward for each action sequence.

        # We start by initializing an array to keep track of the reward for each
        # of these rollouts.
        sum_of_rewards = np.zeros(
            (self.ensemble_size, self.mpc_num_action_sequences), dtype=np.float32
        )
        obs = np.tile(obs, (self.ensemble_size, self.mpc_num_action_sequences, 1))

        # For each action sequence in mpc_horizon (action at time t), we predict the
        # next state for each model in the ensemble, and then compute the reward for
        # each of these rollouts.
        for acs in action_sequences.transpose(1, 0, 2):
            assert acs.shape == (self.mpc_num_action_sequences, self.ac_dim)
            assert obs.shape == (
                self.ensemble_size,
                self.mpc_num_action_sequences,
                self.ob_dim,
            )

            next_obs = []
            for i in range(self.ensemble_size):
                next_obs.append(self.get_dynamics_predictions(i, obs[i], acs))
            next_obs = np.stack(next_obs, axis=0)
            assert next_obs.shape == (
                self.ensemble_size,
                self.mpc_num_action_sequences,
                self.ob_dim,
            )

            acs_repeated = np.expand_dims(acs, axis=0)
            # shape: (ensemble_size, mpc_num_action_sequences, ac_dim)
            acs_repeated = np.repeat(acs_repeated, self.ensemble_size, axis=0)

            rewards = self.get_reward(acs_repeated, next_obs)
            assert rewards.shape == (self.ensemble_size, self.mpc_num_action_sequences)

            sum_of_rewards += rewards
            obs = next_obs

        return sum_of_rewards.mean(axis=0)

    def get_action_sequences(self) -> np.ndarray:
        """
        Generate action sequences for model-predictive control.

        Returns:
            action_sequences: (mpc_num_action_sequences, mpc_horizon, ac_dim)
        """
        raise NotImplementedError

    def get_action(self, obs: np.ndarray):
        """
        Choose the best action using model-predictive control.

        Args:
            obs: (ob_dim,)
        """
        # (mpc_num_action_sequences, mpc_horizon, ac_dim)
        action_sequences = self.get_action_sequences()

        if self.mpc_strategy == "random":
            rewards = self.evaluate_action_sequences(obs, action_sequences)
            assert rewards.shape == (self.mpc_num_action_sequences,)
            best_index = np.argmax(rewards)
            return action_sequences[best_index][0]

        elif self.mpc_strategy == "cem":
            elite_mean, elite_std = None, None
            for i in range(self.cem_num_iters):
                if i == 0:
                    elite_mean = np.zeros((self.mpc_horizon, self.ac_dim))
                    elite_std = np.ones((self.mpc_horizon, self.ac_dim))
                else:
                    action_sequences = np.random.normal(
                        elite_mean, elite_std, size=action_sequences.shape
                    )

                low = -1
                high = 1
                action_sequences = np.clip(action_sequences, low, high)
                rewards = self.evaluate_action_sequences(obs, action_sequences)

                # Select the elites
                elite_indices = np.argsort(rewards)[-self.cem_num_elites :]
                elites = action_sequences[elite_indices]

                # Update the mean and std
                elite_mean = np.mean(elites, axis=0)
                elite_std = np.std(elites, axis=0) + 1e-8

            return elite_mean[0]

        else:
            raise ValueError(f"Invalid MPC strategy '{self.mpc_strategy}'")
