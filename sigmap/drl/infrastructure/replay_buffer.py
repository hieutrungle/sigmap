import numpy as np
import pickle
from typing import Tuple, Union
import os
import json
import glob

# from sigmap.drl.infrastructure.data_types import Observations
from sigmap.utils import utils


class WirelessReplayBuffer:
    def __init__(
        self,
        buffer_size: int = 100000,
        saved_dir="",
        name="wireless_replay_buffer",
        prefix_idx=0,
        seed=0,
    ):
        """
        A replay buffer for wireless environments.

        It is an empty DataBaches object with the ability to insert data.
        """
        super().__init__()
        self.max_size = buffer_size
        self.size_counter = 0
        self.saved_dir = saved_dir
        self.name = name
        self.prefix_idx = prefix_idx
        self.rng = np.random.default_rng(seed)

        self.observations: dict = None
        self.actions: np.ndarray = None
        self.rewards: np.ndarray = None
        self.next_observations: dict = None
        self.dones: np.ndarray = None

    def sample(
        self, batch_size: int
    ) -> Tuple[dict, np.ndarray, np.ndarray, dict, np.ndarray]:

        rand_indices = list(
            self.rng.choice(self.size_counter, size=(batch_size,), replace=False)
            % self.max_size
        )
        observations = {
            "focal_pts": self.observations["focal_pts"][rand_indices],
            "tx_positions": self.observations["tx_positions"][rand_indices],
            "rx_positions": self.observations["rx_positions"][rand_indices],
        }
        next_observations = {
            "focal_pts": self.next_observations["focal_pts"][rand_indices],
            "tx_positions": self.next_observations["tx_positions"][rand_indices],
            "rx_positions": self.next_observations["rx_positions"][rand_indices],
        }
        return {
            "observations": observations,
            "actions": self.actions[rand_indices],
            "rewards": self.rewards[rand_indices],
            "next_observations": next_observations,
            "dones": self.dones[rand_indices],
        }

    def __len__(self):
        return self.size_counter

    def insert(
        self,
        /,
        observation: dict,
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: dict,
        done: np.ndarray,
        is_saved: bool = True,
    ):
        """
        Insert a single transition into the replay buffer.

        Use like:
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
        """

        if self.observations is None:
            self.observations = {
                "focal_pts": np.empty((self.max_size, *observation["focal_pts"].shape)),
                "tx_positions": np.empty(
                    (self.max_size, *observation["tx_positions"].shape)
                ),
                "rx_positions": np.empty(
                    (self.max_size, *observation["rx_positions"].shape)
                ),
            }
            self.actions = np.empty((self.max_size, *action.shape), dtype=action.dtype)
            self.rewards = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)
            self.next_observations = {
                "focal_pts": np.empty(
                    (self.max_size, *next_observation["focal_pts"].shape)
                ),
                "tx_positions": np.empty(
                    (self.max_size, *next_observation["tx_positions"].shape)
                ),
                "rx_positions": np.empty(
                    (self.max_size, *next_observation["rx_positions"].shape)
                ),
            }
            self.dones = np.empty((self.max_size, *done.shape), dtype=done.dtype)

        cur_idx = self.size_counter % self.max_size
        self.observations["focal_pts"][cur_idx] = observation["focal_pts"]
        self.observations["tx_positions"][cur_idx] = observation["tx_positions"]
        self.observations["rx_positions"][cur_idx] = observation["rx_positions"]
        self.actions[cur_idx] = action
        self.rewards[cur_idx] = reward
        self.next_observations["focal_pts"][cur_idx] = next_observation["focal_pts"]
        self.next_observations["tx_positions"][cur_idx] = next_observation[
            "tx_positions"
        ]
        self.next_observations["rx_positions"][cur_idx] = next_observation[
            "rx_positions"
        ]
        self.dones[cur_idx] = done

        # Save the batch to a file
        if is_saved:
            saved_path = os.path.join(
                self.saved_dir, f"{self.name}_{self.prefix_idx:04d}.txt"
            )
            self.save_data_to_file(
                saved_path, observation, action, reward, next_observation, done
            )

        self.size_counter += 1
        if self.size_counter > (self.max_size * (self.prefix_idx + 1)):
            self.prefix_idx += 1

    def save_data_to_file(
        self,
        saved_path: str,
        observation: dict,
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: dict,
        done: np.ndarray,
    ) -> None:
        batch = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "done": done,
        }
        with open(saved_path, "a") as f:
            json.dump(batch, f, cls=utils.NpEncoder)
            f.write("\n")


class ReplayBuffer:
    def __init__(self, buffer_size: int = 1000000):
        self.max_size = buffer_size
        self.size_counter = 0
        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.dones = None

    def sample(self, batch_size: int):
        rand_indices = (
            np.random.randint(0, self.size_counter, size=(batch_size,)) % self.max_size
        )
        return {
            "observations": self.observations[rand_indices],
            "actions": self.actions[rand_indices],
            "rewards": self.rewards[rand_indices],
            "next_observations": self.next_observations[rand_indices],
            "dones": self.dones[rand_indices],
        }

    def __len__(self):
        return self.size_counter

    def insert(
        self,
        /,
        observation: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: np.ndarray,
        done: np.ndarray,
    ):
        """
        Insert a single transition into the replay buffer.

        Use like:
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
        """
        if isinstance(reward, (float, int)):
            reward = np.array(reward)
        if isinstance(done, bool):
            done = np.array(done)
        if isinstance(action, int):
            action = np.array(action, dtype=np.int64)

        if self.observations is None:
            self.observations = np.empty(
                (self.max_size, *observation.shape), dtype=observation.dtype
            )
            self.actions = np.empty((self.max_size, *action.shape), dtype=action.dtype)
            self.rewards = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)
            self.next_observations = np.empty(
                (self.max_size, *next_observation.shape), dtype=next_observation.dtype
            )
            self.dones = np.empty((self.max_size, *done.shape), dtype=done.dtype)

        assert observation.shape == self.observations.shape[1:]
        assert action.shape == self.actions.shape[1:]
        assert reward.shape == ()
        assert next_observation.shape == self.next_observations.shape[1:]
        assert done.shape == ()

        self.observations[self.size_counter % self.max_size] = observation
        self.actions[self.size_counter % self.max_size] = action
        self.rewards[self.size_counter % self.max_size] = reward
        self.next_observations[self.size_counter % self.max_size] = next_observation
        self.dones[self.size_counter % self.max_size] = done

        self.size_counter += 1
        self.size_counter = self.size_counter

    def batched_insert(
        self,
        /,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
    ):
        """
        Insert a batch of transitions into the replay buffer.
        """
        if self.observations is None:
            self.observations = np.empty(
                (self.max_size, *observations.shape[1:]), dtype=observations.dtype
            )
            self.actions = np.empty(
                (self.max_size, *actions.shape[1:]), dtype=actions.dtype
            )
            self.rewards = np.empty(
                (self.max_size, *rewards.shape[1:]), dtype=rewards.dtype
            )
            self.next_observations = np.empty(
                (self.max_size, *next_observations.shape[1:]),
                dtype=next_observations.dtype,
            )
            self.dones = np.empty((self.max_size, *dones.shape[1:]), dtype=dones.dtype)

        assert observations.shape[1:] == self.observations.shape[1:]
        assert actions.shape[1:] == self.actions.shape[1:]
        assert rewards.shape[1:] == self.rewards.shape[1:]
        assert next_observations.shape[1:] == self.next_observations.shape[1:]
        assert dones.shape[1:] == self.dones.shape[1:]

        indices = (
            np.arange(self.size_counter, self.size_counter + observations.shape[0])
            % self.max_size
        )
        self.observations[indices] = observations
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_observations[indices] = next_observations
        self.dones[indices] = dones

        self.size_counter += observations.shape[0]
        self.size_counter = self.size_counter


class MemoryEfficientReplayBuffer:
    """
    A memory-efficient version of the replay buffer for when observations are stacked.
    """

    def __init__(self, frame_history_len: int, capacity=1000000):
        self.max_size = capacity

        # Technically we need max_size*2 to support both obs and next_obs.
        # Otherwise we'll end up overwriting old observations' frames, but the
        # corresponding next_observation_framebuffer_idcs will still point to the old frames.
        # (It's okay though because the unused data will be paged out)
        self.max_framebuffer_size = 2 * capacity

        self.frame_history_len = frame_history_len
        self.size = 0
        self.actions = None
        self.rewards = None
        self.dones = None

        self.observation_framebuffer_idcs = None
        self.next_observation_framebuffer_idcs = None
        self.framebuffer = None
        self.observation_shape = None

        self.current_trajectory_begin = None
        self.current_trajectory_framebuffer_begin = None
        self.framebuffer_idx = None

        self.recent_observation_framebuffer_idcs = None

    def sample(self, batch_size):
        rand_indices = (
            np.random.randint(0, self.size, size=(batch_size,)) % self.max_size
        )

        observation_framebuffer_idcs = (
            self.observation_framebuffer_idcs[rand_indices] % self.max_framebuffer_size
        )
        next_observation_framebuffer_idcs = (
            self.next_observation_framebuffer_idcs[rand_indices]
            % self.max_framebuffer_size
        )

        return {
            "observations": self.framebuffer[observation_framebuffer_idcs],
            "actions": self.actions[rand_indices],
            "rewards": self.rewards[rand_indices],
            "next_observations": self.framebuffer[next_observation_framebuffer_idcs],
            "dones": self.dones[rand_indices],
        }

    def __len__(self):
        return self.size

    def _insert_frame(self, frame: np.ndarray) -> int:
        """
        Insert a single frame into the replay buffer.

        Returns the index of the frame in the replay buffer.
        """
        assert frame.ndim == 2, "Single-frame observation should have dimensions (H, W)"
        assert frame.dtype == np.uint8, "Observation should be uint8 (0-255)"

        self.framebuffer[self.framebuffer_idx] = frame
        frame_idx = self.framebuffer_idx
        self.framebuffer_idx = self.framebuffer_idx + 1

        return frame_idx

    def _compute_frame_history_idcs(
        self, latest_framebuffer_idx: int, trajectory_begin_framebuffer_idx: int
    ) -> np.ndarray:
        """
        Get the indices of the frames in the replay buffer corresponding to the
        frame history for the given latest frame index and trajectory begin index.

        Indices are into the observation buffer, not the regular buffers.
        """
        return np.maximum(
            np.arange(-self.frame_history_len + 1, 1) + latest_framebuffer_idx,
            trajectory_begin_framebuffer_idx,
        )

    def on_reset(
        self,
        /,
        observation: np.ndarray,
    ):
        """
        Call this with the first observation of a new episode.
        """
        assert (
            observation.ndim == 2
        ), "Single-frame observation should have dimensions (H, W)"
        assert observation.dtype == np.uint8, "Observation should be uint8 (0-255)"

        if self.observation_shape is None:
            self.observation_shape = observation.shape
        else:
            assert self.observation_shape == observation.shape

        if self.observation_framebuffer_idcs is None:
            self.observation_framebuffer_idcs = np.empty(
                (self.max_size, self.frame_history_len), dtype=np.int64
            )
            self.next_observation_framebuffer_idcs = np.empty(
                (self.max_size, self.frame_history_len), dtype=np.int64
            )
            self.framebuffer = np.empty(
                (self.max_framebuffer_size, *observation.shape), dtype=observation.dtype
            )
            self.framebuffer_idx = 0
            self.current_trajectory_begin = 0
            self.current_trajectory_framebuffer_begin = 0

        self.current_trajectory_begin = self.size

        # Insert the observation.
        self.current_trajectory_framebuffer_begin = self._insert_frame(observation)
        # Compute, but don't store until we have a next observation.
        self.recent_observation_framebuffer_idcs = self._compute_frame_history_idcs(
            self.current_trajectory_framebuffer_begin,
            self.current_trajectory_framebuffer_begin,
        )

    def insert(
        self,
        /,
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: np.ndarray,
        done: np.ndarray,
    ):
        """
        Insert a single transition into the replay buffer.

        Use like:
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
                truncated=truncated,
            )
        """
        if isinstance(reward, (float, int)):
            reward = np.array(reward)
        if isinstance(done, bool):
            done = np.array(done)
        if isinstance(action, int):
            action = np.array(action, dtype=np.int64)

        assert (
            next_observation.ndim == 2
        ), "Single-frame observation should have dimensions (H, W)"
        assert next_observation.dtype == np.uint8, "Observation should be uint8 (0-255)"

        if self.actions is None:
            self.actions = np.empty((self.max_size, *action.shape), dtype=action.dtype)
            self.rewards = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)
            self.dones = np.empty((self.max_size, *done.shape), dtype=done.dtype)

        assert action.shape == self.actions.shape[1:]
        assert reward.shape == ()
        assert next_observation.shape == self.observation_shape
        assert done.shape == ()

        self.observation_framebuffer_idcs[self.size % self.max_size] = (
            self.recent_observation_framebuffer_idcs
        )
        self.actions[self.size % self.max_size] = action
        self.rewards[self.size % self.max_size] = reward
        self.dones[self.size % self.max_size] = done

        next_frame_idx = self._insert_frame(next_observation)

        # Compute indices for the next observation.
        next_framebuffer_idcs = self._compute_frame_history_idcs(
            next_frame_idx, self.current_trajectory_framebuffer_begin
        )
        self.next_observation_framebuffer_idcs[self.size % self.max_size] = (
            next_framebuffer_idcs
        )

        self.size += 1

        # Set up the observation for the next step.
        # This won't be sampled yet, and it will be overwritten if we start a new episode.
        self.recent_observation_framebuffer_idcs = next_framebuffer_idcs
