from sigmap.utils import utils, timer, logger, map_prep
import os
import sionna.rt


class SignalCoverageMap:
    def __init__(self, args, config):
        self.args = args
        self.config = config

        # input directories
        self.compute_scene_path = args.compute_scene_path
        self.viz_scene_path = args.viz_scene_path

        # output directories
        self.img_dir = utils.get_image_dir(config)

        # Camera
        self.cam = map_prep.prepare_camera(self.config)

    @timer.Timer(
        text="Elapsed coverage map time: {:0.4f} seconds\n", logger_fn=logger.log
    )
    def compute_cmap(self) -> sionna.rt.CoverageMap:
        # Compute coverage maps with ceiling on
        logger.log(f"Computing coverage map for {self.compute_scene_path}")
        scene = map_prep.prepare_scene(self.config, self.compute_scene_path, self.cam)

        cm = scene.coverage_map(
            max_depth=self.config.cm_max_depth,
            cm_cell_size=self.config.cm_cell_size,
            num_samples=self.config.cm_num_samples,
            diffraction=self.config.diffraction,
        )
        return cm

    @timer.Timer(text="Elapsed paths time: {:0.4f} seconds\n", logger_fn=logger.log)
    def compute_paths(self) -> sionna.rt.Paths:
        # Compute coverage maps with ceiling on
        logger.log(f"Computing paths for {self.compute_scene_path}")
        scene = map_prep.prepare_scene(self.config, self.compute_scene_path, self.cam)

        paths = scene.compute_paths(
            max_depth=self.config.path_max_depth,
            num_samples=self.config.path_num_samples,
        )
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

        scene = map_prep.prepare_scene(self.config, self.viz_scene_path, self.cam)

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
