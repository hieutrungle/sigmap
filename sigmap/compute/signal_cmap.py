from sigmap.utils import utils, timer, logger, map_prep
import os
import sionna.rt
from typing import List
import tensorflow as tf


class SignalCoverageMap:
    def __init__(self, args, config):
        self.config = config

        # input directories
        self._compute_scene_path = args.compute_scene_path
        self._viz_scene_path = args.viz_scene_path

        # output directories
        self.img_dir = utils.get_image_dir(config)

        # Camera
        self._cam = map_prep.prepare_camera(self.config)

    @property
    def cam(self):
        return self._cam

    @timer.Timer(
        text="Elapsed coverage map time: {:0.4f} seconds\n", logger_fn=logger.log
    )
    def compute_cmap(self, **kwargs) -> sionna.rt.CoverageMap:
        # Compute coverage maps with ceiling on
        logger.log(f"Computing coverage map for {self._compute_scene_path}")
        scene = map_prep.prepare_scene(self.config, self._compute_scene_path, self.cam)

        cm_kwargs = dict(
            max_depth=self.config.cm_max_depth,
            cm_cell_size=self.config.cm_cell_size,
            num_samples=self.config.cm_num_samples,
            diffraction=self.config.diffraction,
        )
        if kwargs:
            cm_kwargs.update(kwargs)

        cmap = scene.coverage_map(**cm_kwargs)
        return cmap

    @timer.Timer(text="Elapsed paths time: {:0.4f} seconds\n", logger_fn=logger.log)
    def compute_paths(self, **kwargs) -> sionna.rt.Paths:
        # Compute coverage maps with ceiling on
        logger.log(f"Computing paths for {self._compute_scene_path}")
        scene = map_prep.prepare_scene(self.config, self._compute_scene_path, self.cam)

        paths_kwargs = dict(
            max_depth=self.config.path_max_depth,
            num_samples=self.config.path_num_samples,
        )
        if kwargs:
            paths_kwargs.update(kwargs)

        paths = scene.compute_paths(**paths_kwargs)
        return paths

    def compute_render(
        self, cmap_enabled: bool = False, paths_enabled: bool = False
    ) -> None:

        # Visualize coverage maps with ceiling off
        if cmap_enabled:
            cm = self.compute_cmap()
        else:
            cm = None

        if paths_enabled:
            paths = self.compute_paths()
        else:
            paths = None

        scene = map_prep.prepare_scene(self.config, self._viz_scene_path, self.cam)

        render_filename = utils.create_filename(
            self.img_dir, f"{self.config.mitsuba_filename}_00000.png"
        )
        render_config = dict(
            camera=self.cam,
            paths=paths,
            filename=render_filename,
            coverage_map=cm,
            cm_vmin=self.config.cm_vmin,
            cm_vmax=self.config.cm_vmax,
            resolution=self.config.resolution,
            show_devices=True,
        )
        scene.render_to_file(**render_config)

    def render_to_file(
        self,
        coverage_map: sionna.rt.CoverageMap = None,
        paths: sionna.rt.Paths = None,
        filename: str = None,
    ) -> None:
        scene = map_prep.prepare_scene(self.config, self._viz_scene_path, self.cam)

        if filename is None:
            render_filename = utils.create_filename(
                self.img_dir, f"{self.config.mitsuba_filename}_00000.png"
            )
        else:
            render_filename = filename
        render_config = dict(
            camera=self.cam,
            paths=paths,
            filename=render_filename,
            coverage_map=coverage_map,
            cm_vmin=self.config.cm_vmin,
            cm_vmax=self.config.cm_vmax,
            resolution=self.config.resolution,
            show_devices=True,
        )
        scene.render_to_file(**render_config)

    def get_received_power_slow(self, coverage_map: sionna.rt.CoverageMap) -> tf.Tensor:
        coverage_map_tensor = coverage_map.as_tensor()
        coverage_map_centers = coverage_map.cell_centers
        rx_position = self.config.rx_position
        distances = tf.norm(coverage_map_centers - rx_position, axis=-1)
        min_dist = tf.reduce_min(distances)
        min_ind = tf.where(tf.equal(distances, min_dist))[0]

        return coverage_map_tensor[0, min_ind[0], min_ind[1]]

    def get_received_power(self, coverage_map: sionna.rt.CoverageMap) -> tf.Tensor:
        coverage_map_tensor = coverage_map.as_tensor()
        coverage_map_centers = coverage_map.cell_centers
        rx_position = tf.convert_to_tensor(self.config.rx_position)

        top_left_pos = coverage_map_centers[0, 0, 0:2] - (coverage_map.cell_size / 2)
        x_distance_to_top_left = rx_position[0] - top_left_pos[0]
        y_distance_to_top_left = rx_position[1] - top_left_pos[1]

        ind_y = int(y_distance_to_top_left / coverage_map.cell_size[1])
        ind_x = int(x_distance_to_top_left / coverage_map.cell_size[0])

        return coverage_map_tensor[0, ind_y, ind_x]

    def get_viz_scene(self) -> sionna.rt.Scene:
        scene = map_prep.prepare_scene(self.config, self._viz_scene_path, self.cam)
        return scene
