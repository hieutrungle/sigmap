import os
import shutil
import re
import argparse
from sigmap.utils import logger
import glob


def mkdir_not_exists(folder_dir):
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)


def mkdir_with_replacement(folder_dir):
    if os.path.exists(folder_dir):
        shutil.rmtree(folder_dir)
    os.makedirs(folder_dir)


def create_filename(dir, filename):
    """Create a filename in the given directory. If the filename already exists, append a number to the filename."""
    mkdir_not_exists(dir)
    filename = os.path.join(dir, filename)
    tmp_filename = filename
    i = 0
    while os.path.exists(tmp_filename):
        i += 1
        tmp_filename = filename.split(".")[0] + f"_{i:03d}." + filename.split(".")[1]
    filename = tmp_filename
    return filename


# Sorting
def tryint(s):
    try:
        return int(s)
    except:
        return s


# Sorting
def alphanum_key(s):
    """Turn a string into a list of string and number chunks.
    "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split("([0-9]+)", s)]


# Sorting
def sort_nicely(l):
    """Sort the given list in the way that humans expect."""
    l.sort(key=alphanum_key)


class Config:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return str(self.__dict__)


# Default for args
def scene_defaults():
    """
    Defaults for scene.
    """
    return dict(
        filename="",
        cam_position=[-7.5, -4.0, 50.0],
        cam_orientation=[0.0, 0.0, 0.0],
        cam_look_at=[-7.5, -4.0, 0.0],
        frequency=28e9,
        synthetic_array=True,
        scene_end="",
    )


def device_defaults():
    """
    Defaults for devices.
    """
    return dict(
        tx_num_rows=4,
        tx_num_cols=4,
        tx_vertical_spacing=0.5,
        tx_horizontal_spacing=0.5,
        tx_pattern="tr38901",
        tx_polarization="V",
        tx_position=[2.2, 0.0, 1.5],
        tx_orientation=[0.0, 0.0, 0.0],
        rx_included=False,
        rx_num_rows=1,
        rx_num_cols=1,
        rx_vertical_spacing=0.5,
        rx_horizontal_spacing=0.5,
        rx_pattern="iso",
        rx_polarization="V",
        rx_position=[-1.0, -4.225, 1.5],
        rx_orientation=[0.0, 0.0, 0.0],
        device_end="",
    )


def rt_defaults():
    """
    Defaults for ray tracing.
    """
    return dict(
        max_depth=15,
        cm_cell_size=[0.2, 0.2],
        num_samples=6e6,
        diffraction=True,
        rt_end="",
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


# Logging
def log_args(args):
    """Logs arguments to the console."""
    logger.log(f"{'*'*23} ARGS BEGIN {'*'*23}")
    if args.verbose == True:
        message = ""
        for k, v in args.__dict__.items():
            if isinstance(v, str):
                message += f"{k} = '{v}'\n"
            else:
                message += f"{k} = {v}\n"
        logger.log(f"{message}")
    logger.log(f"{'*'*24} ARGS END {'*'*24}\n")


# Input and output folders
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
    sort_nicely(cm_scene_folders)

    viz_scene_folders = glob.glob(
        os.path.join(blender_scene_dir, f"{args.scene_name}_color_*")
    )
    viz_scene_folders = sorted(
        viz_scene_folders, key=lambda x: float(re.findall("(\d+)", x)[0])
    )
    sort_nicely(viz_scene_folders)

    return (cm_scene_folders, viz_scene_folders)


def get_output_folders(args):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    asset_dir = os.path.join(parent_dir, "assets")

    img_folder = os.path.join(asset_dir, "images")
    mkdir_not_exists(img_folder)
    img_tmp_folder = os.path.join(img_folder, f"tmp_{args.scene_name}")
    mkdir_not_exists(img_tmp_folder)
    video_folder = os.path.join(asset_dir, "videos")
    mkdir_not_exists(video_folder)

    return (img_folder, img_tmp_folder, video_folder)
