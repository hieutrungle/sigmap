import os
import re

import numpy as np
from gymnasium.envs.mujoco import mujoco_env
from gymnasium.spaces import Box
from gymnasium import Env, spaces
from utils import utils
from gymnasium.wrappers import TimeLimit
import subprocess


class WirelessEnv(Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }

    def __init__(
        self,
        sionna_config_file,
        num_devices,
        num_tiles_per_device,
        controlled_elements,
        **kwargs,
    ):
        super(WirelessEnv, self).__init__()

        self.sionna_config_file = sionna_config_file
        print(f"Loading Sionna config from {self.sionna_config_file}")

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.ep_return = 0
        self.ep_step = 0

        # Random initial state
        self.device_states = np.random.uniform(
            0, 2 * np.pi, size=self.observation_shape
        )
        self.device_states = np.asarray(self.device_states, dtype=np.float32)
        self.device_states = np.clip(self.device_states, 0, 2 * np.pi)

        self.info = {"episode": {"r": 0, "l": 0}}
        return self.device_states, self.info

    def step(self, action, **kwargs):

        # termination
        terminated = False

        # truncation
        self.ep_step += 1
        truncated = False

        # next observation
        self.device_states = self.device_states + action
        self.device_states = np.clip(self.device_states, 0, 2 * np.pi)
        next_observation = self.device_states

        # reward
        reward = self._cal_reward(self.device_states)

        # info
        self.info.update({"episode": {"r": reward, "l": self.ep_step}})

        return next_observation, reward, terminated, truncated, self.info

    def _cal_reward(self, device_states):
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

        # Config modifications
        config = {}
        config["tx_position"] = [1.0, 0.0, 1.5]
        config["rx_position"] = [-3.0, -4.2, 1.5]
        config["cm_num_samples"] = 2e6
        config["cm_max_depth"] = 15
        config["path_num_samples"] = 3e6
        config["path_max_depth"] = 2
        self._modify_config_file(self.sionna_config_file, **config)

        # Blender export
        blender_app = os.getenv("BLENDER_APP")
        blender_dir = os.getenv("BLENDER_DIR")
        sigmap_dir = os.getenv("SIGMAP_DIR")
        assets_dir = os.getenv("ASSETS_DIR")
        blender_output_dir = os.path.join(assets_dir, "blender")

        blender_script = os.path.join(
            sigmap_dir, "sigmap", "blender_script", "bl_drl.py"
        )
        blender_command = [
            blender_app,
            "-b",
            os.path.join(blender_dir, "models", "simple_hallway_color.blend"),
            "--python",
            blender_script,
            "--",
            "-cfg",
            self.sionna_config_file,
            "-o",
            blender_output_dir,
        ]
        try:
            subprocess.run(blender_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running Blender command: {e}")
            exit(1)

        # Sionna simulation
        scene_name = utils.load_yaml_file(self.sionna_config_file)["scene_name"]
        compute_scene_path = (
            subprocess.check_output(
                f'find {os.path.join(assets_dir, "blender", scene_name)} -type d -name "ceiling_idx*"',
                shell=True,
            )
            .decode()
            .strip()
        )
        compute_scene_path = (
            subprocess.check_output(
                f'find "{compute_scene_path}" -type f -name "*.xml"', shell=True
            )
            .decode()
            .strip()
        )
        viz_scene_path = (
            subprocess.check_output(
                f'find {os.path.join(assets_dir, "blender", scene_name)} -type d -name "idx*"',
                shell=True,
            )
            .decode()
            .strip()
        )
        viz_scene_path = (
            subprocess.check_output(
                f'find "{viz_scene_path}" -type f -name "*.xml"', shell=True
            )
            .decode()
            .strip()
        )
        print(f"\nCompute scene path: {compute_scene_path}")
        print(f"Viz scene path: {viz_scene_path}")

        sionna_command = [
            "python",
            os.path.join(sigmap_dir, "sigmap", "sub_tasks", "run_cmap.py"),
            "-cfg",
            self.sionna_config_file,
            "--compute_scene_path",
            compute_scene_path,
            "--viz_scene_path",
            viz_scene_path,
            "--cmap_enabled",
            "--verbose",
        ]
        try:
            subprocess.run(sionna_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running Sionna command: {e}")
            exit(1)

        return 0.0

    def _modify_config_file(self, config_file, **kwargs):
        config_kwargs = utils.load_yaml_file(config_file)
        for k, v in kwargs.items():
            if isinstance(v, str):
                if v.lower() == "true":
                    config_kwargs[k] = True
                elif v.lower() == "false":
                    config_kwargs[k] = False
                elif v.isnumeric():
                    config_kwargs[k] = float(v)
                elif re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$", v):
                    config_kwargs[k] = float(v)
            config_kwargs[k] = v
        utils.write_yaml_file(config_file, config_kwargs)
