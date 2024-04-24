import numpy as np
from typing import Union


# class Observation:
#     def __init__(
#         self,
#         device_state: Union[np.ndarray, list],
#         tx_position: Union[np.ndarray, list],
#         rx_position: Union[np.ndarray, list],
#     ):
#         if isinstance(device_state, list):
#             device_state = np.array(device_state)
#         if isinstance(tx_position, list):
#             tx_position = np.array(tx_position)
#         if isinstance(rx_position, list):
#             rx_position = np.array(rx_position)

#         self.observation = {
#             "device_state": device_state,
#             "tx_position": tx_position,
#             "rx_position": rx_position,
#         }

#     def __getitem__(self, key: str) -> np.ndarray:
#         if key not in ["device_state", "tx_position", "rx_position"]:
#             raise ValueError("Key not found")
#         return self.observation[key]


# class Observations:
#     def __init__(self, observations: list[Observation]):
#         device_states = []
#         tx_positions = []
#         rx_positions = []
#         for observation in observations:
#             device_state = np.expand_dims(observation["device_state"], axis=0)
#             tx_position = np.expand_dims(observation["tx_position"], axis=0)
#             rx_position = np.expand_dims(observation["rx_position"], axis=0)
#             device_states.append(device_state)
#             tx_positions.append(tx_position)
#             rx_positions.append(rx_position)

#         device_states = np.concatenate(device_states, axis=0)
#         tx_positions = np.concatenate(tx_positions, axis=0)
#         rx_positions = np.concatenate(rx_positions, axis=0)

#         self.observations = {
#             "device_state": device_states,
#             "tx_position": tx_positions,
#             "rx_position": rx_positions,
#         }

#     def __getitem__(self, key: str) -> np.ndarray:
#         if key not in ["device_state", "tx_position", "rx_position"]:
#             raise ValueError("Key not found")
#         return self.observations[key]

#     def __len__(self):
#         return self.observations["device_state"].shape[0]

#     def append(self, observation: Observation):
#         device_state = np.expand_dims(observation["device_state"], axis=0)
#         tx_position = np.expand_dims(observation["tx_position"], axis=0)
#         rx_position = np.expand_dims(observation["rx_position"], axis=0)

#         self.observations["device_state"] = np.concatenate(
#             [self.observations["device_state"], device_state], axis=0
#         )
#         self.observations["tx_position"] = np.concatenate(
#             [self.observations["tx_position"], tx_position], axis=0
#         )
#         self.observations["rx_position"] = np.concatenate(
#             [self.observations["rx_position"], rx_position], axis=0
#         )

# class _Observation:
#     """
#     Class to store the observation for each device in the environment

#     Args:
#         device_state (Union[np.ndarray, list]): [batch_size, num_devices, num_tiles, num_features]
#         tx_position (Union[np.ndarray, list]): [batch_size, num_tx, 3]
#         rx_position (Union[np.ndarray, list]): [batch_size, num_rx, 3]

#     Returns:
#         Observations: [batch_size, num_devices, num_tiles, num_features], [batch_size, num_tx, 3], [batch_size, num_rx, 3]
#     """

#     def __init__(
#         self,
#         device_state: Union[np.ndarray, list] = None,
#         tx_position: Union[np.ndarray, list] = None,
#         rx_position: Union[np.ndarray, list] = None,
#     ):
#         """
#         Args:
#             device_state (Union[np.ndarray, list]): [batch_size, num_devices, num_tiles, num_features]
#             tx_position (Union[np.ndarray, list]): [batch_size, num_tx, 3]
#             rx_position (Union[np.ndarray, list]): [batch_size, num_rx, 3]
#         """
#         if isinstance(device_state, list):
#             device_state = np.array(device_state)
#         if isinstance(tx_position, list):
#             tx_position = np.array(tx_position)
#         if isinstance(rx_position, list):
#             rx_position = np.array(rx_position)
#         self.device_state = device_state
#         self.tx_position = tx_position
#         self.rx_position = rx_position

#     # def __getitem__(self, key: str) -> np.ndarray:
#     #     if key not in ["device_state", "tx_position", "rx_position"]:
#     #         raise ValueError("Key not found")
#     #     return getattr(self, key)

#     def __getitem__(self, idxs: list) -> Observations:
#         device_state = self.device_state[idxs]
#         tx_position = self.tx_position[idxs]
#         rx_position = self.rx_position[idxs]
#         return Observations(device_state, tx_position, rx_position)

#     def __len__(self):
#         return self.device_state.shape[0]

#     def append(
#         self, device_state: np.ndarray, tx_position: np.ndarray, rx_position: np.ndarray
#     ):
#         self.device_state = np.concatenate([self.device_state, device_state], axis=0)
#         self.tx_position = np.concatenate([self.tx_position, tx_position], axis=0)
#         self.rx_position = np.concatenate([self.rx_position, rx_position], axis=0)


class Observation:
    """
    Class to store the observation for each device in the environment

    Args:
        device_state (Union[np.ndarray, list]): [num_devices, num_tiles, num_features]
        tx_position (Union[np.ndarray, list]): [num_tx, 3]
        rx_position (Union[np.ndarray, list]): [num_rx, 3]

    Returns:
        Observation: [num_devices, num_tiles, num_features], [num_tx, 3], [num_rx, 3]
    """

    def __init__(
        self,
        device_state: Union[np.ndarray, list],
        tx_position: Union[np.ndarray, list],
        rx_position: Union[np.ndarray, list],
    ):
        if isinstance(device_state, list):
            device_state = np.array(device_state)
        if isinstance(tx_position, list):
            tx_position = np.array(tx_position)
        if isinstance(rx_position, list):
            rx_position = np.array(rx_position)

        self.device_state = device_state
        self.tx_position = tx_position
        self.rx_position = rx_position

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in ["device_state", "tx_position", "rx_position"]:
            raise ValueError("Key not found")
        return getattr(self, key)


class Observations:
    """
    Class to store a batch of observations

    Args:
        device_state (np.ndarray): [batch_size, num_devices, num_tiles, num_features]
        tx_position (np.ndarray): [batch_size, num_tx, 3]
        rx_position (np.ndarray): [batch_size, num_rx, 3]

    Returns:
        Observations: [batch_size, num_devices, num_tiles, num_features], [batch_size, num_tx, 3], [batch_size, num_rx, 3]
    """

    def __init__(
        self,
        device_state: np.ndarray,
        tx_position: np.ndarray,
        rx_position: np.ndarray,
        size: int = None,
    ):
        """
        Args:
            device_state (np.ndarray): [batch_size, num_devices, num_tiles, num_features]
            tx_position (np.ndarray): [batch_size, num_tx, 3]
            rx_position (np.ndarray): [batch_size, num_rx, 3]
            size (int): Number of observations

        If size is not None, initialize Observations with zeros. All variables are initialized with zeros.
        Otherwise, use the input values.
        """

        assert (
            len(device_state.shape) == 4
        ), f"device_state should have 4 dimensions: [batch_size, num_devices, num_tiles, num_features]. Got {device_state.shape}"
        assert (
            len(tx_position.shape) == 3
        ), f"tx_position should have 3 dimensions: [batch_size, num_tx, 3]. Got {tx_position.shape}"
        assert (
            len(rx_position.shape) == 3
        ), f"rx_position should have 3 dimensions: [batch_size, num_rx, 3]. Got {rx_position.shape}"

        self.device_state_shape = device_state.shape[1:]
        self.tx_position_shape = tx_position.shape[1:]
        self.rx_position_shape = rx_position.shape[1:]

        if size is not None:
            self._initialize(size)
        else:
            self.device_state = device_state
            self.tx_position = tx_position
            self.rx_position = rx_position

    def _initialize(self, size: int):
        self.device_state = np.zeros((size, *self.device_state_shape))
        self.tx_position = np.zeros((size, *self.tx_position_shape))
        self.rx_position = np.zeros((size, *self.rx_position_shape))

    def __getitem__(self, idxs: Union[int, list]):
        if isinstance(idxs, int):
            idxs = [idxs]
        if not isinstance(idxs, list):
            raise ValueError("idxs should be a list of int or int")
        device_state = self.device_state[idxs]
        tx_position = self.tx_position[idxs]
        rx_position = self.rx_position[idxs]
        return Observations(device_state, tx_position, rx_position)

    def __setitem__(self, idxs: Union[int, list], observations):
        if isinstance(idxs, int):
            idxs = [idxs]
        if not isinstance(idxs, list):
            raise ValueError(f"idxs should be a list of int or int. Got {type(idxs)}")
        if not isinstance(observations, Observations):
            raise ValueError(
                f"value should be an instance of Observations. Got {type(observations)}"
            )
        self.device_state[idxs] = observations.device_state
        self.tx_position[idxs] = observations.tx_position
        self.rx_position[idxs] = observations.rx_position

    def __len__(self):
        return self.size

    def append(
        self, device_state: np.ndarray, tx_position: np.ndarray, rx_position: np.ndarray
    ):
        assert len(device_state.shape) == len(
            self.device_state.shape
        ), f"Invalid shape. Expected [batch_size, {self.device_state.shape[1:]}]. Got {device_state.shape}"
        assert len(tx_position.shape) == len(
            self.tx_position.shape
        ), f"Invalid shape. Expected [batch_size, {self.tx_position.shape[1:]}]. Got {tx_position.shape}"
        assert len(rx_position.shape) == len(
            self.rx_position.shape
        ), f"Invalid shape. Expected [batch_size, {self.rx_position.shape[1:]}]. Got {rx_position.shape}"

        self.device_state = np.concatenate([self.device_state, device_state], axis=0)
        self.tx_position = np.concatenate([self.tx_position, tx_position], axis=0)
        self.rx_position = np.concatenate([self.rx_position, rx_position], axis=0)

    # def __str__(self):
    #     return (
    #         f"Observations: {self.device_state}, {self.tx_position}, {self.rx_position}"
    #     )


# class Observations:
#     """
#     Class to store a batch of observations

#     Args:
#         device_state (Union[np.ndarray, list]): [batch_size, num_devices, num_tiles, num_features]
#         tx_position (Union[np.ndarray, list]): [batch_size, num_tx, 3]
#         rx_position (Union[np.ndarray, list]): [batch_size, num_rx, 3]

#     Returns:
#         Observations: [batch_size, num_devices, num_tiles, num_features], [batch_size, num_tx, 3], [batch_size, num_rx, 3]
#     """

#     def __init__(
#         self,
#         device_state: Union[np.ndarray, list] = None,
#         tx_position: Union[np.ndarray, list] = None,
#         rx_position: Union[np.ndarray, list] = None,
#         max_size: int = None,
#     ):
#         """
#         Args:
#             device_state (Union[np.ndarray, list]): [batch_size, num_devices, num_tiles, num_features]
#             tx_position (Union[np.ndarray, list]): [batch_size, num_tx, 3]
#             rx_position (Union[np.ndarray, list]): [batch_size, num_rx, 3]
#         """
#         if isinstance(device_state, list):
#             device_state = np.array(device_state)
#         if isinstance(tx_position, list):
#             tx_position = np.array(tx_position)
#         if isinstance(rx_position, list):
#             rx_position = np.array(rx_position)

#         if len(device_state.shape) != 4:
#             device_state = np.expand_dims(device_state, axis=0)
#         if len(tx_position.shape) != 3:
#             tx_position = np.expand_dims(tx_position, axis=0)
#         if len(rx_position.shape) != 3:
#             rx_position = np.expand_dims(rx_position, axis=0)

#         assert (
#             len(device_state.shape) == 4
#         ), f"device_state should have 4 dimensions. Got {device_state.shape}"
#         assert (
#             len(tx_position.shape) == 3
#         ), f"tx_position should have 3 dimensions. Got {tx_position.shape}"
#         assert (
#             len(rx_position.shape) == 3
#         ), f"rx_position should have 3 dimensions. Got {rx_position.shape}"

#         self.device_state = device_state
#         self.tx_position = tx_position
#         self.rx_position = rx_position

#         if max_size is not None:
#             self._initialize(max_size)

#     def __getitem__(self, idxs: Union[int, list]):
#         if isinstance(idxs, int):
#             idxs = [idxs]
#         if not isinstance(idxs, list):
#             raise ValueError("idxs should be a list of int or int")
#         device_state = self.device_state[idxs]
#         tx_position = self.tx_position[idxs]
#         rx_position = self.rx_position[idxs]
#         return Observations(device_state, tx_position, rx_position)

#     def __setitem__(self, idxs: Union[int, list], observations):
#         if isinstance(idxs, int):
#             idxs = [idxs]
#         if not isinstance(idxs, list):
#             raise ValueError(f"idxs should be a list of int or int. Got {type(idxs)}")
#         if not isinstance(observations, Observations):
#             raise ValueError(
#                 f"value should be an instance of Observations. Got {type(observations)}"
#             )
#         self.device_state[idxs] = observations.device_state
#         self.tx_position[idxs] = observations.tx_position
#         self.rx_position[idxs] = observations.rx_position

#     def __len__(self):
#         return self.device_state.shape[0]

#     def append(
#         self, device_state: np.ndarray, tx_position: np.ndarray, rx_position: np.ndarray
#     ):
#         assert len(device_state.shape) == len(
#             self.device_state.shape
#         ), f"Invalid shape. Expected [batch_size, {self.device_state.shape[1:]}]. Got {device_state.shape}"
#         assert len(tx_position.shape) == len(
#             self.tx_position.shape
#         ), f"Invalid shape. Expected [batch_size, {self.tx_position.shape[1:]}]. Got {tx_position.shape}"
#         assert len(rx_position.shape) == len(
#             self.rx_position.shape
#         ), f"Invalid shape. Expected [batch_size, {self.rx_position.shape[1:]}]. Got {rx_position.shape}"

#         self.device_state = np.concatenate([self.device_state, device_state], axis=0)
#         self.tx_position = np.concatenate([self.tx_position, tx_position], axis=0)
#         self.rx_position = np.concatenate([self.rx_position, rx_position], axis=0)

#     def _initialize(self, max_size: int):
#         self.device_state = np.zeros((max_size, *self.device_state.shape[1:]))
#         self.tx_position = np.zeros((max_size, *self.tx_position.shape[1:]))
#         self.rx_position = np.zeros((max_size, *self.rx_position.shape[1:]))
