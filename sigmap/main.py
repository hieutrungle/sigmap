import time
import os
import glob
import re
import subprocess
import argparse

gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import sionna
from sigmap.utils import logger, utils, timer, map_prep

# Import Sionna RT components
from sionna.rt import (
    load_scene,
    Transmitter,
    Receiver,
    PlanarArray,
    Camera,
    Paths,
    RadioMaterial,
    LambertianPattern,
)

# For link-level simulations
from sionna.channel import (
    cir_to_ofdm_channel,
    subcarrier_frequencies,
    OFDMChannel,
    ApplyOFDMChannel,
    CIRDataset,
)
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.utils import compute_ber, ebnodb2no, PlotBER
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement

from sigmap.utils import utils, timer, logger


def main():
    args = create_argparser().parse_args()
    log_dir = "./tmp_logs"
    utils.mkdir_not_exists(log_dir)
    logger.configure(dir=log_dir)
    utils.log_args(args)

    # Prepare folders
    cm_scene_folders, viz_scene_folders = get_input_folders(args)
    img_folder, img_tmp_folder, video_folder = get_output_folders(args)

    for i, (cm_scene_folder, viz_scene_folder) in enumerate(
        zip(cm_scene_folders, viz_scene_folders)
    ):
        # Compute coverage maps with ceiling on
        cam = map_prep.prepare_camera(args)
        args.filename = os.path.join(cm_scene_folder, f"{args.filename}.xml")
        scene = map_prep.prepare_scene(args, cam)

        cm = scene.coverage_map(
            max_depth=args.max_depth,
            cm_cell_size=args.cm_cell_size,
            num_samples=args.num_samples,
            diffraction=args.diffraction,
        )
        # cm=None

        # Visualize coverage maps
        args.filename = os.path.join(viz_scene_folder, f"{args.filename}.xml")
        scene = map_prep.prepare_scene(args, cam)

        scene.render_to_file(
            camera=cam,
            filename=os.path.join(img_tmp_folder, f"{args.filename}_{i:02d}.png"),
            coverage_map=cm,
            cm_vmin=args.cm_vmin,
            cm_vmax=args.cm_vmax,
            show_devices=False,
            resolution=args.resolution,
        )

        if i == 0:
            scene.render_to_file(
                cam,
                filename=os.path.join(img_folder, f"{args.filename}_scene_{i:02d}.png"),
                cm_vmin=args.cm_vmin,
                cm_vmax=args.cm_vmax,
                show_devices=True,
                resolution=args.resolution,
            )
            scene.render(
                cam,
                coverage_map=cm,
                cm_vmin=args.cm_vmin,
                cm_vmax=args.cm_vmax,
                show_devices=True,
                resolution=args.resolution,
            )


def create_argparser():
    """Parses command line arguments."""
    defaults = dict(
        filename="hallway",
        scene_name="tee_hallway",
        resolution=(1920, 1080),
        cm_vmin=-150,
        cm_vmax=-70,
        verbose=True,
        default_end="",
    )
    defaults.update(utils.scene_defaults())
    defaults.update(utils.device_defaults())
    defaults.update(utils.rt_defaults())
    parser = argparse.ArgumentParser()
    utils.add_dict_to_argparser(parser, defaults)
    return parser


def get_input_folders(args):
    # Scene directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    asset_dir = os.path.join(parent_dir, "assets")
    blender_scene_dir = os.path.join(asset_dir, "blender")

    cm_scene_folders = glob.glob(
        os.path.join(blender_scene_dir, f"{args.scene_name}_ceiling_color_*")
    )
    cm_scene_folders = sorted(
        cm_scene_folders, key=lambda x: float(re.findall("(\d+)", x)[0])
    )
    utils.sort_nicely(cm_scene_folders)

    viz_scene_folders = glob.glob(
        os.path.join(blender_scene_dir, f"{args.scene_name}_color_*")
    )
    viz_scene_folders = sorted(
        viz_scene_folders, key=lambda x: float(re.findall("(\d+)", x)[0])
    )
    utils.sort_nicely(viz_scene_folders)

    return (cm_scene_folders, viz_scene_folders)


def get_output_folders(args):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    asset_dir = os.path.join(parent_dir, "assets")

    img_folder = os.path.join(asset_dir, "images")
    utils.mkdir_not_exists(img_folder)
    img_tmp_folder = os.path.join(img_folder, f"tmp_{args.scene_name}")
    utils.mkdir_not_exists(img_tmp_folder)
    video_folder = os.path.join(asset_dir, "videos")
    utils.mkdir_not_exists(video_folder)

    return (img_folder, img_tmp_folder, video_folder)


if __name__ == "__main__":
    main()
