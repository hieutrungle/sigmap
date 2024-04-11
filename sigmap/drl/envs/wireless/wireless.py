import os
import re
import subprocess
import pickle
import json

import numpy as np
from gymnasium import Env, spaces
from sigmap.utils import utils


class WirelessEnv(Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 10,
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

        self.num_devices = num_devices
        self.num_tiles_per_device = num_tiles_per_device
        self.controlled_elements = controlled_elements

        self.sionna_config_file = sionna_config_file
        config_kwargs = utils.load_yaml_file(sionna_config_file)
        self._default_tx_position = config_kwargs["tx_position"]
        self._default_rx_position = config_kwargs["rx_position"]

        # Observation space
        self.device_state_shape = (
            self.num_devices,
            self.num_tiles_per_device,
            self.controlled_elements,
        )

        # Rotation is in radians, [0, 2pi]
        device_state_space = spaces.Box(
            low=0, high=2 * np.pi, shape=self.device_state_shape, dtype=np.float32
        )
        tx_position_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        rx_position_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "device_state": device_state_space,
                "tx_position": tx_position_space,
                "rx_position": rx_position_space,
            }
        )

        # Action outputs the angle delta of the beamforming vector
        # next_obs = obs + delta (action)
        self.action_space = spaces.Box(
            low=-np.pi / 4,
            high=np.pi / 4,
            shape=self.device_state_shape,
            dtype=np.float32,
        )

        # State of all devices
        self._device_state = None
        self._tx_position = None
        self._rx_position = None
        self.info = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.ep_return = 0
        self.ep_step = 0

        # Random initial state
        self._device_state = np.random.uniform(
            0, 2 * np.pi, size=self.device_state_shape
        )
        self._device_state = np.asarray(self._device_state, dtype=np.float32)
        self._device_state = np.clip(self._device_state, 0, 2 * np.pi)

        self._tx_position = self._default_tx_position
        self._rx_position = self._default_rx_position

        self.info = {"episode": {"r": 0, "l": 0}}
        self.info.update(
            {"tx_position": self._tx_position, "rx_position": self._rx_position}
        )
        return self._get_obs(), self.info

    def _get_obs(self):
        return {
            "device_state": self._device_state,
            "tx_position": self._tx_position,
            "rx_position": self._rx_position,
        }

    def step(self, action, **kwargs):

        # termination
        terminated = False

        # truncation
        self.ep_step += 1
        truncated = False

        # reward
        ## Save device_state to a tmp file
        ## Open Blender to read the file and assign values to devices' tiles
        ## Then export the geometry file to Sionna
        reward = self._cal_reward(self._device_state)

        # next observation
        self._device_state = self._device_state + action
        self._device_state = np.clip(self._device_state, 0, 2 * np.pi)
        next_observation = self._get_obs()

        # info
        self.info.update({"episode": {"r": reward, "l": self.ep_step}})

        return next_observation, reward, terminated, truncated, self.info

    def _cal_reward(self, device_state):
        """
        Reward function for the wireless environment.

        Args:
            device_state (np.ndarray): States of all devices. Shape: (num_devices, num_tiles_per_device, controlled_elements).

        Returns:
            float: Reward value.

        This function calculates the reward for the wireless environment based on the given device states.
        The device states represent the rotation angle (in radians) of all devices' tiles.

        The reward calculation involves two steps:
        1. The device states are used to generate a geometry file using Blender.
        2. The generated geometry file is then fed into a Sionna-based program to calculate the path gain, which is transformed into the reward.

        Before the reward calculation, the configuration file is modified to set the transmitter and receiver positions, as well as other parameters.

        Note: This function assumes that the necessary environment variables (BLENDER_APP, BLENDER_DIR, SIGMAP_DIR, ASSETS_DIR, TMP_DIR) are properly set.

        """

        # Config modifications
        config = {}
        config["tx_position"] = [1.0, 0.0, 1.5]
        config["rx_position"] = [-3.0, -4.2, 1.5]
        config["cm_num_samples"] = 1e6
        config["cm_max_depth"] = 15
        config["path_num_samples"] = 1e6
        config["path_max_depth"] = 2
        self._modify_config_file(self.sionna_config_file, **config)

        # Generate geometry file
        self._run_blender(device_state)

        # Run Sionna to get reward
        reward = self._run_sionna()

        return reward

    def _run_blender(self, device_state):
        # Blender export

        blender_app = utils.get_os_dir("BLENDER_APP")
        blender_dir = utils.get_os_dir("BLENDER_DIR")
        sigmap_dir = utils.get_os_dir("SIGMAP_DIR")
        assets_dir = utils.get_os_dir("ASSETS_DIR")
        blender_output_dir = os.path.join(assets_dir, "blender")
        tmp_dir = utils.get_tmp_dir()

        tmp_file = os.path.join(tmp_dir, "device_state.pkl")
        with open(tmp_file, "wb") as f:
            pickle.dump(device_state, f)

        blender_script = os.path.join(
            sigmap_dir, "sigmap", "blender_script", "bl_drl.py"
        )
        bl_output_txt = os.path.join(tmp_dir, "bl_outputs.txt")
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
            subprocess.run(blender_command, check=True, stdout=open(bl_output_txt, "a"))
        except subprocess.CalledProcessError as e:
            os.remove(tmp_file)
            raise Exception(f"Error running Blender command: {e}")
        finally:
            os.remove(tmp_file)

    def _run_sionna(self) -> float:
        path_gain = self._cal_path_gain()
        return path_gain

    def _cal_path_gain(self) -> float:
        sigmap_dir = utils.get_os_dir("SIGMAP_DIR")
        assets_dir = utils.get_assets_dir()
        tmp_dir = utils.get_tmp_dir()

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

        sionna_output_txt = os.path.join(tmp_dir, "sionna_outputs.txt")
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
            # "--verbose",
        ]
        try:
            subprocess.run(
                sionna_command, check=True, stdout=open(sionna_output_txt, "a")
            )
        except subprocess.CalledProcessError as e:
            raise Exception(f"Error running Sionna command: {e}")

        results_file = os.path.join(tmp_dir, "path_gain.txt")
        with open(results_file, "r") as f:
            results_dict = json.load(f)
        path_gain = results_dict["path_gain"]
        return path_gain

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
