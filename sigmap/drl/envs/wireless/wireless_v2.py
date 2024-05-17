import os
import re
import subprocess
import pickle
from typing import Tuple

import numpy as np
from gymnasium import Env, spaces
from sigmap.utils import utils
import sigmap
import time
import matplotlib.pyplot as plt
import json


class WirelessEnvV2(Env):
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
        action_scale=1.0,
        **kwargs,
    ):
        super(WirelessEnvV2, self).__init__()

        self.current_time = "_" + time.strftime("%d-%m-%Y_%H-%M-%S")

        self.action_scale = action_scale
        self.num_devices = num_devices
        self.num_tiles_per_device = num_tiles_per_device
        self.controlled_elements = controlled_elements

        self.sionna_config_file = sionna_config_file
        config_kwargs = utils.load_yaml_file(sionna_config_file)
        self._default_tx_positions = np.array(config_kwargs["tx_position"])
        self._default_rx_positions = np.array(config_kwargs["rx_position"])
        # TODO: Now we only have 1 TX and 1 RX, need to update this for multiple TX and RX
        self._default_tx_positions = np.expand_dims(self._default_tx_positions, axis=0)
        self._default_rx_positions = np.expand_dims(self._default_rx_positions, axis=0)

        # Observation space
        # Each device has 2 focal points
        self.focal_pts_shape = (num_devices, 2, 3)
        low = np.array([-20, -20, -5])
        low = np.tile(low, (self.num_devices, 2, 1))
        self.focal_pts_low = low
        high = np.array([20, 20, 5])
        high = np.tile(high, (self.num_devices, 2, 1))
        self.focal_pts_high = high
        focal_pts_space = spaces.Box(
            self.focal_pts_low,
            self.focal_pts_high,
            shape=self.focal_pts_shape,
            dtype=np.float32,
        )
        tx_positions_space = spaces.Box(
            -np.inf, np.inf, shape=self._default_tx_positions.shape, dtype=np.float32
        )
        rx_positions_space = spaces.Box(
            -np.inf, np.inf, shape=self._default_rx_positions.shape, dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {
                "focal_pts": focal_pts_space,
                "tx_positions": tx_positions_space,
                "rx_positions": rx_positions_space,
            }
        )

        # Action space
        # represent the delta x, y, z of the focal points
        self.action_space = spaces.Box(
            -self.action_scale,
            self.action_scale,
            shape=self.focal_pts_shape,
            dtype=np.float32,
        )

        # State of all devices
        self._focal_pts = None
        self._tx_positions = None
        self._rx_positions = None
        self.info = None

    def _get_obs(self) -> dict:
        observation = {
            "focal_pts": np.asarray(self._focal_pts, dtype=np.float32),
            "tx_positions": np.asarray(self._tx_positions, dtype=np.float32),
            "rx_positions": np.asarray(self._rx_positions, dtype=np.float32),
        }
        return observation

    def reset(self, seed=None, options=None) -> Tuple[dict, dict]:
        super().reset(seed=seed, options=options)
        self.ep_return = 0
        self.ep_step = 0

        # Random initial state
        # self._focal_pts = np.random.uniform(
        #     self.focal_pts_low, self.focal_pts_high, size=self.focal_pts_shape
        # )
        self._focal_pts = np.random.randn(*self.focal_pts_shape)
        self._focal_pts[:, 0] += self._default_tx_positions
        self._focal_pts[:, 1] += self._default_rx_positions
        self._focal_pts = np.asarray(self._focal_pts, dtype=np.float32)
        self._focal_pts = np.clip(
            self._focal_pts, self.focal_pts_low, self.focal_pts_high
        )

        self._tx_positions = np.asarray(self._default_tx_positions, dtype=np.float32)
        self._rx_positions = np.asarray(self._default_rx_positions, dtype=np.float32)

        self.info = {"episode": {"r": 0, "l": 0}}
        self.info.update(
            {"tx_positions": self._tx_positions, "rx_positions": self._rx_positions}
        )
        return self._get_obs(), self.info

    def step(
        self, action: np.ndarray, **kwargs
    ) -> Tuple[dict, float, bool, bool, dict]:

        # next observation
        self._focal_pts = self._focal_pts + action
        self._focal_pts = np.clip(
            self._focal_pts, self.focal_pts_low, self.focal_pts_high
        )
        next_observation = self._get_obs()

        # termination
        # Check if self._focal_pts is out of bounds
        terminated = False
        if np.any(self._focal_pts < self.focal_pts_low) or np.any(
            self._focal_pts > self.focal_pts_high
        ):
            terminated = True

        # truncation
        self.ep_step += 1
        truncated = False

        # reward
        ## Save focal_pts to a tmp file
        ## Open Blender to read the file and assign values to devices' tiles
        ## Then export the geometry file to Sionna
        reward = self._cal_reward(self._focal_pts)

        # info
        self.info.update({"episode": {"r": reward, "l": self.ep_step}})

        return next_observation, reward, terminated, truncated, self.info

    def _cal_reward(self, focal_pts):
        """
        Reward function for the wireless environment.

        Args:
            focal_pts (np.ndarray): States of all devices. Shape: (num_devices, num_tiles_per_device, controlled_elements).

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
        # config["tx_position"] = [1.0, 0.0, 1.5]
        # config["rx_position"] = [-11.0, -4.2, 1.5]
        # config["cm_num_samples"] = 1e6
        # config["cm_max_depth"] = 15
        # config["path_num_samples"] = 1e6
        # config["path_max_depth"] = 2
        self._modify_config_file(self.sionna_config_file, **config)

        # Generate geometry file
        self._run_blender(focal_pts)

        # Run Sionna to get reward
        reward = self._run_sionna()

        return reward

    def _run_blender(self, focal_pts):
        # Blender export

        blender_app = utils.get_os_dir("BLENDER_APP")
        blender_dir = utils.get_os_dir("BLENDER_DIR")
        sigmap_dir = utils.get_os_dir("SIGMAP_DIR")
        assets_dir = utils.get_os_dir("ASSETS_DIR")
        blender_output_dir = os.path.join(assets_dir, "blender")
        tmp_dir = utils.get_tmp_dir()

        tmp_file = os.path.join(tmp_dir, "focal_pts.pkl")
        with open(tmp_file, "wb") as f:
            pickle.dump(focal_pts, f)

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
        assets_dir = utils.get_assets_dir()

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

        sigmap_dir = utils.get_os_dir("SIGMAP_DIR")
        siona_script = os.path.join(sigmap_dir, "sigmap", "sub_tasks", "run_cmap.py")
        img_dir = os.path.join(assets_dir, "images", scene_name + self.current_time)
        mitsuba_filename = utils.load_yaml_file(self.sionna_config_file)[
            "mitsuba_filename"
        ]
        render_filename = utils.create_filename(
            img_dir, f"{mitsuba_filename}_00000.png"
        )
        sionna_command = [
            "python",
            siona_script,
            "-cfg",
            self.sionna_config_file,
            "--compute_scene_path",
            compute_scene_path,
            "--viz_scene_path",
            viz_scene_path,
            "--saved_path",
            render_filename,
            "--cmap_enabled",
        ]
        tmp_dir = utils.get_tmp_dir()
        sionna_output_txt = os.path.join(tmp_dir, "sionna_outputs.txt")
        try:
            subprocess.run(
                sionna_command, check=True, stdout=open(sionna_output_txt, "a")
            )
        except subprocess.CalledProcessError as e:
            raise Exception(f"Error running Blender command: {e}")
        finally:
            pass

        results_file = os.path.join(tmp_dir, "path_gain.txt")
        with open(results_file, "r") as f:
            results_dict = json.load(f)
            path_gain = results_dict["path_gain"]

        # config = sigmap.utils.scripting_utils.make_sionna_config(
        #     self.sionna_config_file
        # )
        # sig_cmap = sigmap.compute.signal_cmap.SignalCoverageMap(
        #     config, compute_scene_path, viz_scene_path
        # )
        # coverage_map = sig_cmap.compute_cmap()

        # img_dir = os.path.join(assets_dir, "images", scene_name + self.current_time)
        # mitsuba_filename = utils.load_yaml_file(self.sionna_config_file)[
        #     "mitsuba_filename"
        # ]
        # render_filename = utils.create_filename(
        #     img_dir, f"{mitsuba_filename}_00000.png"
        # )
        # sig_cmap.render_to_file(coverage_map, filename=render_filename)
        # path_gain = sig_cmap.get_path_gain(
        #     coverage_map,
        # )
        # del coverage_map
        # del sig_cmap

        path_gain = float(path_gain)
        path_gain_dB = utils.linear2dB(path_gain)
        plt.clf()
        plt.close("all")
        return path_gain_dB

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
