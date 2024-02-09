from sigmap.utils import utils, timer, logger, map_prep
import os
import sionna.rt


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

    def get_viz_scene(self) -> sionna.rt.Scene:
        scene = map_prep.prepare_scene(self.config, self._viz_scene_path, self.cam)
        return scene
