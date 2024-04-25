import numpy as np
import pickle
from typing import Tuple, Union
import os
import json
import glob

from sigmap.drl.infrastructure.data_type import Observation, Observations
from sigmap.utils import utils


class DataBatch:
    def __init__(
        self,
        observation: Observation = None,
        action: np.ndarray = None,
        reward: np.ndarray = None,
        next_observation: Observation = None,
        done: np.ndarray = None,
    ):
        """
        Construct a DataBatch object from the given data.
        """
        self.batch = None
        if (
            observation is not None
            and action is not None
            and reward is not None
            and next_observation is not None
            and done is not None
        ):
            self._set_batch(observation, action, reward, next_observation, done)

    def _set_batch(
        self,
        observation: Observation,
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: Observation,
        done: np.ndarray,
    ):
        observation = self._convert_to_python_type(observation)
        action = action.tolist() if isinstance(action, np.ndarray) else action
        reward = reward.tolist() if isinstance(reward, np.ndarray) else reward
        next_observation = self._convert_to_python_type(next_observation)
        done = done.tolist() if isinstance(done, np.ndarray) else done

        self.batch = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "done": done,
        }

    def _convert_to_python_type(self, data: dict):
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
        return data

    def save(self, file_path: str):
        with open(file_path, "a") as f:
            json.dump(self.batch, f)
            f.write("\n")

    def load_first_entry(self, file_path: str):
        """
        Load the first batch from the file only if the batch is not already predefined
        """
        if self.batch is None:
            self.batch = self.read_first_line(file_path)
            self.batch = json.loads(self.batch)

    def load_last_entry(self, file_path: str):
        """
        Load the last batch from the file only if the batch is not already predefined
        """
        if self.batch is None:
            self.batch = self.read_last_line(file_path)
            self.batch = json.loads(self.batch)

    def read_first_line(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            first_line = f.readline().decode()
        return first_line

    def read_last_line(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            try:  # catch OSError in case of a one line file
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b"\n":
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode()
        return last_line

    def read_n_to_last_line(self, filename, n=1) -> str:
        """Returns the nth before last line of a file (n=1 gives last line)"""
        num_newlines = 0
        with open(filename, "rb") as f:
            try:
                f.seek(-2, os.SEEK_END)
                while num_newlines < n:
                    f.seek(-2, os.SEEK_CUR)
                    if f.read(1) == b"\n":
                        num_newlines += 1
            except OSError:
                f.seek(0)
            n_to_last_line = f.readline().decode()

        return n_to_last_line

    def __str__(self):
        return str(self.batch)

    def __repr__(self):
        return str(self.batch)

    def __getitem__(self, key) -> Union[dict, np.ndarray]:
        """
        Get the value of the key in the batch.
        Permissible keys are "next_observation", "action", "reward", "observation", and "done".
        """
        if key not in [
            "next_observation",
            "action",
            "reward",
            "observation",
            "done",
        ]:
            raise ValueError("Key not found")
        return self.batch[key]

    def __setitem__(self, key, value):
        if key == "next_observation" or key == "observation":
            value = self._convert_to_python_type(value)
        elif key == "action" or key == "reward" or key == "done":
            value = value.tolist() if isinstance(value, np.ndarray) else value
        else:
            raise ValueError("Key not found")
        self.batch[key] = value

    def __len__(self):
        return len(self.batch)


class DataBatches:
    def __init__(
        self,
        observations: list[dict[Union[np.ndarray, list]]] = None,
        actions: list[np.ndarray] = None,
        rewards: list[np.ndarray] = None,
        next_observations: list[dict[Union[np.ndarray, list]]] = None,
        dones: list[np.ndarray] = None,
    ):
        """
        Construct a list of DataBatch objects from the given data.
        """
        self.batches = []
        if (
            observations is not None
            and actions is not None
            and rewards is not None
            and next_observations is not None
            and dones is not None
        ):
            self._set_batches(observations, actions, rewards, next_observations, dones)

    def _set_batches(
        self,
        observations: list[dict[Union[np.ndarray, list]]],
        actions: list[np.ndarray],
        rewards: list[np.ndarray],
        next_observations: list[dict[Union[np.ndarray, list]]],
        dones: list[np.ndarray],
    ):
        self.batches = []
        for i in range(len(next_observations)):
            self.batches.append(
                DataBatch(
                    observations[i],
                    actions[i],
                    rewards[i],
                    next_observations[i],
                    dones[i],
                )
            )

    def save(self, file_path: str):
        for i, batch in enumerate(self.batches):
            batch.save(file_path)

    def load(self, file_path: str):
        with open(file_path, "r") as f:
            for line in f:
                self.batches.append(json.loads(line))

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx: int) -> DataBatch:
        return self.batches[idx]

    def __setitem__(self, idx, value):
        self.batches[idx] = value

    def __iter__(self):
        return iter(self.batches)


class WirelessReplayBuffer:
    def __init__(
        self,
        buffer_size: int = 100000,
        saved_dir="",
        name="wireless_replay_buffer",
        prefix_idx=0,
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

        self.observations: Observations = None
        self.actions: np.ndarray = None
        self.rewards: np.ndarray = None
        self.next_observations: Observations = None
        self.dones: np.ndarray = None

    def sample(self, batch_size: int) -> list[DataBatch]:

        rand_indices = list(
            np.random.randint(0, self.size_counter, size=(batch_size,)) % self.max_size
        )
        return (
            self.observations[rand_indices],
            self.actions[rand_indices],
            self.rewards[rand_indices],
            self.next_observations[rand_indices],
            self.dones[rand_indices],
        )

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
            self.observations = Observations(
                focal_pts=observation["focal_pts"][None],
                tx_position=observation["tx_position"][None],
                rx_position=observation["rx_position"][None],
                size=self.max_size,
            )
            self.actions = np.empty((self.max_size, *action.shape), dtype=action.dtype)
            self.rewards = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)
            self.next_observations = Observations(
                focal_pts=next_observation["focal_pts"][None],
                tx_position=next_observation["tx_position"][None],
                rx_position=next_observation["rx_position"][None],
                size=self.max_size,
            )
            self.dones = np.empty((self.max_size, *done.shape), dtype=done.dtype)

        observations = Observations(
            focal_pts=observation["focal_pts"][None],
            tx_position=observation["tx_position"][None],
            rx_position=observation["rx_position"][None],
        )
        next_observations = Observations(
            focal_pts=next_observation["focal_pts"][None],
            tx_position=next_observation["tx_position"][None],
            rx_position=next_observation["rx_position"][None],
        )
        cur_idx = self.size_counter % self.max_size
        self.observations[cur_idx] = observations
        self.actions[cur_idx] = action
        self.rewards[cur_idx] = reward
        self.next_observations[cur_idx] = next_observations
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

    # def batched_insert(
    #     self,
    #     /,
    #     observations: list[dict[Union[np.ndarray, list]]],
    #     actions: list[np.ndarray],
    #     rewards: list[np.ndarray],
    #     next_observations: list[dict[Union[np.ndarray, list]]],
    #     dones: list[np.ndarray],
    #     is_saved: bool = True,
    # ) -> None:
    #     """
    #     Insert a batch of transitions into the replay buffer.
    #     """
    #     batches = DataBatches(
    #         observations=observations,
    #         actions=actions,
    #         rewards=rewards,
    #         next_observations=next_observations,
    #         dones=dones,
    #     )

    #     # If the replay buffer is empty, fill it with the first batch
    #     # This prevent allocating memory for future batches
    #     if len(self.batches) == 0:
    #         self.batches = [batches[0] for _ in range(self.max_size)]

    #     indices = (
    #         np.arange(self.size_counter, self.size_counter + len(actions))
    #         % self.max_size
    #     )
    #     for i, target_idx in enumerate(indices):
    #         self.batches[target_idx] = batches[i]

    #     if is_saved:
    #         saved_path = os.path.join(
    #             self.saved_dir, f"{self.name}_{self.prefix_idx:04d}.txt"
    #         )
    #         batches.save(saved_path)

    #     # Increment index for saving to a new file
    #     self.size_counter += len(actions)
    #     if self.size_counter > (self.max_size * (self.prefix_idx + 1)):
    #         self.prefix_idx += 1

    # def load_replay_buffer(self):
    #     filepaths = glob.glob(os.path.join(self.saved_dir, f"{self.name}_*.txt"))
    #     for file in filepaths:
    #         batches = DataBatches()
    #         batches.load(file)

    #         for batch in batches:
    #             self.insert(
    #                 observation=batch["observation"],
    #                 action=batch["action"],
    #                 reward=batch["reward"],
    #                 next_observation=batch["next_observation"],
    #                 done=batch["done"],
    #                 is_saved=False,
    #             )


# class WirelessReplayBuffer(DataBatches):
#     def __init__(
#         self,
#         buffer_size: int = 100000,
#         saved_dir="",
#         name="wireless_replay_buffer",
#         prefix_idx=0,
#     ):
#         """
#         A replay buffer for wireless environments.

#         It is an empty DataBaches object with the ability to insert data.
#         """
#         super().__init__()
#         self.max_size = buffer_size
#         self.size_counter = 0
#         self.batches: list[DataBatch] = []
#         self.saved_dir = saved_dir
#         self.name = name
#         self.prefix_idx = prefix_idx

#     def sample(self, batch_size: int) -> list[DataBatch]:

#         rand_indices = (
#             np.random.randint(0, self.size_counter, size=(batch_size,)) % self.max_size
#         )
#         batches = []
#         for rand_idx in rand_indices:
#             batches.append(self.batches[rand_idx])
#         return batches

#     def __len__(self):
#         return self.size_counter

#     def insert(
#         self,
#         /,
#         observation: dict[Union[np.ndarray, list]],
#         action: np.ndarray,
#         reward: np.ndarray,
#         next_observation: dict[Union[np.ndarray, list]],
#         done: np.ndarray,
#         is_saved: bool = True,
#     ):
#         """
#         Insert a single transition into the replay buffer.

#         Use like:
#             replay_buffer.insert(
#                 observation=observation,
#                 action=action,
#                 reward=reward,
#                 next_observation=next_observation,
#                 done=done,
#             )
#         """

#         batch = DataBatch(
#             observation=observation,
#             action=action,
#             reward=reward,
#             next_observation=next_observation,
#             done=done,
#         )

#         # If the replay buffer is empty, fill it with the first batch
#         # This prevent allocating memory for future batches
#         if len(self.batches) == 0:
#             self.batches = [batch for _ in range(self.max_size)]

#         cur_idx = self.size_counter % self.max_size
#         self.batches[cur_idx] = batch

#         if is_saved:
#             saved_path = os.path.join(
#                 self.saved_dir, f"{self.name}_{self.prefix_idx:04d}.txt"
#             )
#             batch.save(saved_path)

#         self.size_counter += 1
#         if self.size_counter > (self.max_size * (self.prefix_idx + 1)):
#             self.prefix_idx += 1

#     def batched_insert(
#         self,
#         /,
#         observations: list[dict[Union[np.ndarray, list]]],
#         actions: list[np.ndarray],
#         rewards: list[np.ndarray],
#         next_observations: list[dict[Union[np.ndarray, list]]],
#         dones: list[np.ndarray],
#         is_saved: bool = True,
#     ) -> None:
#         """
#         Insert a batch of transitions into the replay buffer.
#         """
#         batches = DataBatches(
#             observations=observations,
#             actions=actions,
#             rewards=rewards,
#             next_observations=next_observations,
#             dones=dones,
#         )

#         # If the replay buffer is empty, fill it with the first batch
#         # This prevent allocating memory for future batches
#         if len(self.batches) == 0:
#             self.batches = [batches[0] for _ in range(self.max_size)]

#         indices = (
#             np.arange(self.size_counter, self.size_counter + len(actions))
#             % self.max_size
#         )
#         for i, target_idx in enumerate(indices):
#             self.batches[target_idx] = batches[i]

#         if is_saved:
#             saved_path = os.path.join(
#                 self.saved_dir, f"{self.name}_{self.prefix_idx:04d}.txt"
#             )
#             batches.save(saved_path)

#         # Increment index for saving to a new file
#         self.size_counter += len(actions)
#         if self.size_counter > (self.max_size * (self.prefix_idx + 1)):
#             self.prefix_idx += 1

#     def load_replay_buffer(self):
#         filepaths = glob.glob(os.path.join(self.saved_dir, f"{self.name}_*.txt"))
#         for file in filepaths:
#             batches = DataBatches()
#             batches.load(file)

#             for batch in batches:
#                 self.insert(
#                     observation=batch["observation"],
#                     action=batch["action"],
#                     reward=batch["reward"],
#                     next_observation=batch["next_observation"],
#                     done=batch["done"],
#                     is_saved=False,
#                 )


# class WirelessReplayBuffer:
#     def __init__(self, buffer_size: int = 1000000, saved_dir=""):
#         self.max_size = buffer_size
#         self.saved_dir = saved_dir
#         self.size_counter = 0
#         self.observations = None
#         self.actions = None
#         self.rewards = None
#         self.next_observations = None
#         self.dones = None

#         self.prefix_idx = 0

#     def sample(self, batch_size: int):
#         rand_indices = (
#             np.random.randint(0, self.size_counter, size=(batch_size,)) % self.max_size
#         )
#         return {
#             "observations": self.observations[rand_indices],
#             "actions": self.actions[rand_indices],
#             "rewards": self.rewards[rand_indices],
#             "next_observations": self.next_observations[rand_indices],
#             "dones": self.dones[rand_indices],
#         }

#     def __len__(self):
#         return self.size_counter

#     def insert(
#         self,
#         /,
#         observation: np.ndarray,
#         action: np.ndarray,
#         reward: np.ndarray,
#         next_observation: np.ndarray,
#         done: np.ndarray,
#     ):
#         """
#         Insert a single transition into the replay buffer.

#         Use like:
#             replay_buffer.insert(
#                 observation=observation,
#                 action=action,
#                 reward=reward,
#                 next_observation=next_observation,
#                 done=done,
#             )
#         """

#         if isinstance(reward, (float, int)):
#             reward = np.array(reward)
#         if isinstance(done, bool):
#             done = np.array(done)
#         if isinstance(action, int):
#             action = np.array(action, dtype=np.int64)

#         if self.observations is None:
#             self.observations = [0 for _ in range(self.max_size)]
#             self.actions = np.empty((self.max_size, *action.shape), dtype=action.dtype)
#             self.rewards = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)
#             self.next_observations = [0 for _ in range(self.max_size)]
#             self.dones = np.empty((self.max_size, *done.shape), dtype=done.dtype)

#         assert action.shape == self.actions.shape[1:]
#         assert reward.shape == ()
#         assert done.shape == ()

#         cur_idx = self.size_counter % self.max_size

#         self.observations[cur_idx] = observation
#         self.actions[cur_idx] = action
#         self.rewards[cur_idx] = reward
#         self.next_observations[cur_idx] = next_observation
#         self.dones[cur_idx] = done

#         self.size_counter += 1

#         batch = {
#             "observations": [observation],
#             "actions": np.array([action]),
#             "rewards": np.array([reward]),
#             "next_observations": [next_observation],
#             "dones": np.array([done]),
#         }
#         if self.saved_dir != "":
#             self._save_data_to_file(batch)

#     def batched_insert(
#         self,
#         /,
#         observations: np.ndarray,
#         actions: np.ndarray,
#         rewards: np.ndarray,
#         next_observations: np.ndarray,
#         dones: np.ndarray,
#     ) -> None:
#         """
#         Insert a batch of transitions into the replay buffer.
#         """
#         if self.observations is None:
#             self.observations = [0 for _ in range(self.max_size)]
#             self.actions = np.empty(
#                 (self.max_size, *actions.shape[1:]), dtype=actions.dtype
#             )
#             self.rewards = np.empty(
#                 (self.max_size, *rewards.shape[1:]), dtype=rewards.dtype
#             )
#             self.next_observations = [0 for _ in range(self.max_size)]
#             self.dones = np.empty((self.max_size, *dones.shape[1:]), dtype=dones.dtype)

#         assert actions.shape[1:] == self.actions.shape[1:]
#         assert rewards.shape[1:] == self.rewards.shape[1:]
#         assert dones.shape[1:] == self.dones.shape[1:]

#         indices = (
#             np.arange(self.size_counter, self.size_counter + observations.shape[0])
#             % self.max_size
#         )
#         self.observations[indices] = observations
#         self.actions[indices] = actions
#         self.rewards[indices] = rewards
#         self.next_observations[indices] = next_observations
#         self.dones[indices] = dones

#         self.size_counter += actions.shape[0]

#         batches = {
#             "observations": observations,
#             "actions": actions,
#             "rewards": rewards,
#             "next_observations": next_observations,
#             "dones": dones,
#         }

#         if self.saved_dir != "":
#             self._save_data_to_file(batches)

#     def _save_data_to_file(self, data: dict) -> None:
#         """
#         Save data to a file.
#         """
#         num_batches = data["rewards"].shape[0]
#         if self.size_counter + num_batches > self.max_size:
#             self.prefix_idx += 1
#         file_path = f"{self.saved_dir}/replay_buffer_{self.prefix_idx:04d}.pkl"
#         with open(file_path, "a") as file:
#             for batch in zip(*data.values()):
#                 pickle.dump(batch, file)

#     def save_replay_buffer(self) -> None:
#         """
#         Save the replay buffer to a file.
#         """
#         # TODO: Implement this method.
#         self._store_data_to_file(
#             {
#                 "observations": self.observations,
#                 "actions": self.actions,
#                 "rewards": self.rewards,
#                 "next_observations": self.next_observations,
#                 "dones": self.dones,
#             }
#         )


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
