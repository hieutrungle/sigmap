import os
import argparse

gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sigmap.utils import utils, logger, scripting_utils
from sigmap import compute
import tensorflow as tf
import json
import sionna


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

    tf.random.set_seed(args.seed)

    # Prepare folders
    sig_cmap = compute.signal_cmap.SignalCoverageMap(
        config, args.compute_scene_path, args.viz_scene_path, args.verbose
    )

    # coverage_map = sig_cmap.compute_cmap() if args.cmap_enabled else None
    # path_gain = sig_cmap.get_path_gain(coverage_map)
    # print(f"Path gain from cmap: {path_gain}")
    # print(f"Path gain from cmap: {utils.linear2dB(path_gain)} dB\n")

    # sig_cmap.render_to_file(coverage_map, None, filename=args.saved_path)

    # del coverage_map

    paths = sig_cmap.compute_paths() if args.paths_enabled else None
    subcarrier_spacing = 15e3
    fft_size = 48
    a, tau = paths.cir()
    frequencies = sionna.channel.subcarrier_frequencies(fft_size, subcarrier_spacing)
    h_freq = sionna.channel.cir_to_ofdm_channel(frequencies, a, tau, normalize=False)
    h_freq_avg_power = tf.reduce_mean(tf.abs(h_freq) ** 2).numpy()
    path_gain = h_freq_avg_power
    tmp_dir = utils.get_tmp_dir()
    results_file = os.path.join(tmp_dir, "path_gain.txt")
    results_dict = {
        "path_gain": path_gain,
    }
    with open(results_file, "w") as f:
        json.dump(results_dict, f, cls=utils.NpEncoder)


def create_args() -> argparse.ArgumentParser:
    """Parses command line arguments."""
    defaults = dict()
    # defaults.update(utils.rt_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--compute_scene_path", "-cp", type=str, required=True)
    parser.add_argument("--viz_scene_path", "-vp", type=str)
    parser.add_argument("--saved_path", type=str, default=None)
    parser.add_argument("--cmap_enabled", action="store_true", default=False)
    parser.add_argument("--paths_enabled", action="store_true", default=False)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)

    scripting_utils.add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()
    if args.viz_scene_path is None or args.viz_scene_path == "":
        args.viz_scene_path = args.compute_scene_path
    return args


if __name__ == "__main__":
    main()
