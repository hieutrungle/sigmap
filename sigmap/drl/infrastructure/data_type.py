import numpy as np
from typing import Union


class Observation:
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

        self.observation = {
            "device_state": device_state,
            "tx_position": tx_position,
            "rx_position": rx_position,
        }

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in ["device_state", "tx_position", "rx_position"]:
            raise ValueError("Key not found")
        return self.observation[key]


class Observations:
    def __init__(self, observations: list[Observation]):
        device_states = []
        tx_positions = []
        rx_positions = []
        for observation in observations:
            device_state = np.expand_dims(observation["device_state"], axis=0)
            tx_position = np.expand_dims(observation["tx_position"], axis=0)
            rx_position = np.expand_dims(observation["rx_position"], axis=0)
            device_states.append(device_state)
            tx_positions.append(tx_position)
            rx_positions.append(rx_position)

        device_states = np.concatenate(device_states, axis=0)
        tx_positions = np.concatenate(tx_positions, axis=0)
        rx_positions = np.concatenate(rx_positions, axis=0)

        self.observations = {
            "device_state": device_states,
            "tx_position": tx_positions,
            "rx_position": rx_positions,
        }

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in ["device_state", "tx_position", "rx_position"]:
            raise ValueError("Key not found")
        return self.observations[key]

    def __len__(self):
        return self.observations["device_state"].shape[0]

    def append(self, observation: Observation):
        device_state = np.expand_dims(observation["device_state"], axis=0)
        tx_position = np.expand_dims(observation["tx_position"], axis=0)
        rx_position = np.expand_dims(observation["rx_position"], axis=0)

        self.observations["device_state"] = np.concatenate(
            [self.observations["device_state"], device_state], axis=0
        )
        self.observations["tx_position"] = np.concatenate(
            [self.observations["tx_position"], tx_position], axis=0
        )
        self.observations["rx_position"] = np.concatenate(
            [self.observations["rx_position"], rx_position], axis=0
        )
