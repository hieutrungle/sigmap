import os
import argparse

gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sigmap.utils import utils, logger, scripting_utils
from sigmap import compute
import tensorflow as tf


def main():
    args = create_args()
    config = scripting_utils.make_sionna_config(args.config_file)
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
    coverage_map = sig_cmap.compute_cmap() if args.cmap_enabled else None
    paths = sig_cmap.compute_paths() if args.paths_enabled else None
    sig_cmap.render_to_file(coverage_map, paths)

    # Compute received power
    received_power = sig_cmap.get_received_power(coverage_map)
    results_dir = utils.get_results_dir(config)
    results_file = os.path.join(results_dir, config.scene_name + "_received_power.csv")
    rx_position_str = str(config.rx_position)
    # remove [] from the string
    rx_position_str = str(rx_position_str).replace("[", "").replace("]", "")
    results_dict = {
        str(rx_position_str): received_power.numpy(),
    }
    with open(results_file, "a") as f:
        f.write(utils.dict_to_csv(results_dict))


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
