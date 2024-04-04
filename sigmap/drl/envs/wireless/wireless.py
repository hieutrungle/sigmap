import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from gymnasium.spaces import Box
from gymnasium import Env, spaces


class WirelessEnv(Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }

    def __init__(
        self, num_devices, num_tiles_per_device, controlled_elements, **kwargs
    ):
        super(WirelessEnv, self).__init__()

        self.num_devices = num_devices
        self.num_tiles_per_device = num_tiles_per_device
        self.controlled_elements = controlled_elements
        self.observation_shape = (
            self.num_devices,
            self.num_tiles_per_device,
            self.controlled_elements,
        )

        # Rotation is in radians, [0, 2pi]
        self.observation_space = spaces.Box(
            low=0, high=2 * np.pi, shape=self.observation_shape, dtype=np.float32
        )

        # Action outputs the angle delta of the beamforming vector
        # next_obs = obs + delta (action)
        self.action_space = spaces.Box(
            low=-np.pi / 4,
            high=np.pi / 4,
            shape=self.observation_shape,
            dtype=np.float32,
        )

        # State of all devices
        self.device_states = None
        self.info = None

    def reset(self):
        self.ep_return = 0
        self.ep_step = 0

        # Random initial state,
        self.device_states = (np.random.randn(*self.observation_shape) * 2 * np.pi) % (
            2 * np.pi
        )

        self.info = {"episode": {"r": 0, "l": 0}}
        return self.device_states, self.info

    def step(self, action):

        # termination
        terminated = False

        # truncation
        self.ep_step += 1
        truncated = False
        if self.ep_step >= self.spec.max_episode_steps:
            truncated = True

        # reward (from Blender + Sionna)
        # reward = self.reward_fn(self.device_states)
        reward = self.cal_reward(self.device_states)

        # next observation
        self.device_states = (self.device_states + action) % (2 * np.pi)
        next_observation = self.device_states

        # info
        self.info.update({"episode": {"r": reward, "l": self.ep_step}})

        return next_observation, reward, terminated, truncated, self.info

    def cal_reward(self, device_states):
        """
        Reward function for the wireless environment.

        Args:
            device_states (np.ndarray): States of all devices. Shape: (num_devices, num_tiles_per_device, controlled_elements).

        Returns:
            float: Reward value.

        This uses both Blender and Sionna for the reward calculation.
        Device states are the rotation angle (in radians) of all devices' tiles.

        the variable `device_states` is fed into Blender to get a geometry file.
        The geometry file is then fed into Sionna-based program to produce the path gain, which is transformed into the reward.
        """
        return 0.0
