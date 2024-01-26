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
import subprocess


def main():
    args = create_argparser().parse_args()
    log_dir = "./tmp_logs"
    utils.mkdir_not_exists(log_dir)
    logger.configure(dir=log_dir)
    utils.log_args(args)

    # Prepare folders
    img_folder, img_tmp_folder, video_folder = utils.get_output_folders(args)
    compute_coverage_map(args)

    with timer.Timer(
        text="Elapsed video creation time: {:0.4f} seconds\n", logger_fn=logger.log
    ):
        logger.log(f"\nCreating video for {args.scene_name}")
        create_video(img_tmp_folder, video_folder, args)


def create_argparser() -> argparse.ArgumentParser:
    """Parses command line arguments."""
    defaults = dict(
        blender_filename="hallway",
        scene_name="tee_hallway",
        resolution=(1920, 1080),
        cm_vmin=-150,
        cm_vmax=-70,
        verbose=True,
        video=False,
        default_end="",
    )
    defaults.update(utils.scene_defaults())
    defaults.update(utils.device_defaults())
    defaults.update(utils.rt_defaults())
    parser = argparse.ArgumentParser()
    utils.add_dict_to_argparser(parser, defaults)
    return parser


@timer.Timer(text="Elapsed coverage map time: {:0.4f} seconds\n", logger_fn=logger.log)
def compute_coverage_map(args):
    # Prepare folders
    cm_scene_folders, viz_scene_folders = utils.get_input_folders(args)
    img_folder, img_tmp_folder, video_folder = utils.get_output_folders(args)

    # Compute coverage maps
    for i, (cm_scene_folder, viz_scene_folder) in enumerate(
        zip(cm_scene_folders, viz_scene_folders)
    ):
        # Compute coverage maps with ceiling on
        cam = map_prep.prepare_camera(args)
        filename = os.path.join(cm_scene_folder, f"{args.blender_filename}.xml")
        logger.log(f"Computing coverage map for {filename}")
        scene = map_prep.prepare_scene(args, filename, cam)

        cm = scene.coverage_map(
            max_depth=args.max_depth,
            cm_cell_size=args.cm_cell_size,
            num_samples=args.num_samples,
            diffraction=args.diffraction,
        )

        # Visualize coverage maps
        filename = os.path.join(viz_scene_folder, f"{args.blender_filename}.xml")
        scene = map_prep.prepare_scene(args, filename, cam)

        render_args = dict(
            camera=cam,
            filename=os.path.join(
                img_tmp_folder, f"{args.blender_filename}_{i:02d}.png"
            ),
            coverage_map=cm,
            cm_vmin=args.cm_vmin,
            cm_vmax=args.cm_vmax,
            resolution=args.resolution,
            show_devices=True,
        )
        scene.render_to_file(**render_args)

        if i == 0:
            render_args["filename"] = os.path.join(
                img_folder, f"{args.blender_filename}_scene_{i:02d}.png"
            )
            scene.render_to_file(**render_args)


def check_ffmpeg_installed():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def create_video(img_tmp_folder, video_folder, args):
    if check_ffmpeg_installed():
        video_path = utils.create_filename(video_folder, f"{args.scene_name}.mp4")
        subprocess.call(
            [
                "ffmpeg",
                "-framerate",
                "1",
                "-i",
                os.path.join(img_tmp_folder, f"{args.blender_filename}" + "_%02d.png"),
                "-r",
                "30",
                "-pix_fmt",
                "yuv420p",
                video_path,
            ]
        )
        logger.log(f"Video saved to {video_path}")
    else:
        logger.log("ffmpeg is not installed. Please install ffmpeg to create videos.")


if __name__ == "__main__":
    main()
