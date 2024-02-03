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
    args = create_argparser().parse_args()
    config = scripting_utils.make_conf(args.config_file)
    log_dir = "./tmp_logs"
    utils.mkdir_not_exists(log_dir)
    logger.configure(dir=log_dir)

    logger.log(f"using tensorflow version: {tf.__version__}")
    logger.log(f"is_gpu_available: {tf.config.list_physical_devices('GPU')}")

    if args.verbose:
        utils.log_args(args)
        utils.log_config(config)

    # Prepare folders
    img_folder, img_tmp_folder, video_folder = utils.get_output_folders(config)
    compute.coverage_map.compute_coverage_map(args, config)

    # Create video
    if args.video_enabled:
        logger.log(f"\nCreating video for {config.scene_name}")
        with timer.Timer(
            text="Elapsed video creation time: {:0.4f} seconds\n", logger_fn=logger.log
        ):
            video_gen.create_video(img_tmp_folder, video_folder, config)


def create_argparser() -> argparse.ArgumentParser:
    """Parses command line arguments."""
    defaults = dict()
    # defaults.update(utils.rt_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--video_enabled", action="store_true", default=False)
    parser.add_argument("--index", type=int, default=0)
    scripting_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
