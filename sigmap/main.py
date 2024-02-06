import time
import os
import glob
import re
import subprocess
import argparse

gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import numpy as np
# import tensorflow as tf
# import sionna
from sigmap.utils import logger, utils, timer, map_prep

# # Import Sionna RT components
# from sionna.rt import (
#     load_scene,
#     Transmitter,
#     Receiver,
#     PlanarArray,
#     Camera,
#     Paths,
#     RadioMaterial,
#     LambertianPattern,
# )

# # For link-level simulations
# from sionna.channel import (
#     cir_to_ofdm_channel,
#     subcarrier_frequencies,
#     OFDMChannel,
#     ApplyOFDMChannel,
#     CIRDataset,
# )
# from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
# from sionna.utils import compute_ber, ebnodb2no, PlotBER
# from sionna.ofdm import KBestDetector, LinearDetector
# from sionna.mimo import StreamManagement

from sigmap.utils import utils, timer, logger, scripting_utils, video_gen
from sigmap import compute
import subprocess
import tensorflow as tf


def main():
    args = create_args()
    config = scripting_utils.make_conf(args.config_file)
    log_dir = "./tmp_logs"
    utils.mkdir_not_exists(log_dir)
    logger.configure(dir=log_dir)

    logger.log(f"using tensorflow version: {tf.__version__}")
    if tf.config.list_physical_devices("GPU") == []:
        logger.log(f"no GPU available\n")
    else:
        logger.log(f"Available GPUs: {tf.config.list_physical_devices('GPU')}\n")

    if args.verbose:
        utils.log_args(args)
        utils.log_config(config)

    # Prepare folders
    sig_cmap = compute.signal_cmap.SignalCoverageMap(args, config)
    sig_cmap.compute_render(
        cmap_enabled=args.cmap_enabled, paths_enabled=args.paths_enabled
    )

    # Create video
    if args.video_enabled:
        logger.log(f"\nCreating video for {config.scene_name}")
        with timer.Timer(
            text="Elapsed video creation time: {:0.4f} seconds\n", logger_fn=logger.log
        ):
            img_dir = utils.get_image_dir(config)
            video_dir = utils.get_video_dir(config)
            video_gen.create_video(img_dir, video_dir, config)


def create_args() -> argparse.ArgumentParser:
    """Parses command line arguments."""
    defaults = dict()
    # defaults.update(utils.rt_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--compute_scene_path", "-cp", type=str, required=True)
    parser.add_argument("--viz_scene_path", "-vp", type=str)
    parser.add_argument("--cmap_enabled", action="store_true", default=False)
    parser.add_argument("--paths_enabled", action="store_true", default=False)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--video_enabled", action="store_true", default=False)
    scripting_utils.add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()
    if args.viz_scene_path is None or args.viz_scene_path == "":
        args.viz_scene_path = args.compute_scene_path
    return args


if __name__ == "__main__":
    main()
