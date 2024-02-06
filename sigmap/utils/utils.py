import os
import shutil
import re
import argparse
from sigmap.utils import logger
import glob
from typing import Dict, List, Union, Tuple


def mkdir_not_exists(folder_dir: str) -> None:
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)


def mkdir_with_replacement(folder_dir: str) -> None:
    if os.path.exists(folder_dir):
        shutil.rmtree(folder_dir)
    os.makedirs(folder_dir)


def create_filename(dir: str, filename: str) -> str:
    """Create a filename in the given directory. If the filename already exists, append a number to the filename."""
    mkdir_not_exists(dir)
    filename = os.path.join(dir, filename)
    tmp_filename = filename
    i = 0
    while os.path.exists(tmp_filename):
        i += 1
        ext = filename.rpartition(".")[2]
        name = filename.rpartition(".")[0]
        name = re.sub(r"\d+$", "", name)
        while name[-1] == "_":
            name = name[:-1]
        tmp_filename = name + f"_{i:05d}." + ext
    filename = tmp_filename
    return filename


# Sorting
def tryint(s: str) -> int:
    try:
        return int(s)
    except:
        return s


# Sorting
def alphanum_key(s: str) -> List[Union[str, int]]:
    """Turn a string into a list of string and number chunks.
    "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split("([0-9]+)", s)]


# Sorting
def sort_nicely(l: List[str]):
    """Sort the given list in the way that humans expect."""
    l.sort(key=alphanum_key)


# Logging
def log_args(args: argparse.Namespace) -> None:
    """Logs arguments to the console."""
    logger.log(f"{'*'*23} ARGS BEGIN {'*'*23}")
    message = ""
    for k, v in args.__dict__.items():
        if isinstance(v, str):
            message += f"{k} = '{v}'\n"
        else:
            message += f"{k} = {v}\n"
    logger.log(f"{message}")
    logger.log(f"{'*'*24} ARGS END {'*'*24}\n")


def log_config(config: Dict[str, Union[str, float, bool]]) -> None:
    """Logs configuration to the console."""
    logger.log(f"{'*'*23} CONFIG BEGIN {'*'*23}")
    message = ""
    for k, v in config.__dict__.items():
        if isinstance(v, str):
            message += f"{k} = '{v}'\n"
        else:
            message += f"{k} = {v}\n"
    logger.log(f"{message}")
    logger.log(f"{'*'*24} CONFIG END {'*'*24}\n")


# Input and output folders
def get_input_folders(
    args: argparse.Namespace, config: Dict[str, Union[str, float, bool]]
) -> Tuple[str]:
    # Scene directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    asset_dir = os.path.join(parent_dir, "assets")
    mkdir_not_exists(asset_dir)
    blender_scene_dir = os.path.join(asset_dir, "blender")
    mkdir_not_exists(blender_scene_dir)

    cm_scene_folders = glob.glob(
        os.path.join(
            blender_scene_dir,
            f"{config.scene_name}",
            f"ceiling_idx_{args.index}*",
        )
    )
    cm_scene_folders = sorted(
        cm_scene_folders, key=lambda x: float(re.findall("(\d+)", x)[0])
    )
    sort_nicely(cm_scene_folders)

    viz_scene_folders = glob.glob(
        os.path.join(
            blender_scene_dir,
            f"{config.scene_name}",
            f"idx_{args.index}*",
        )
    )
    viz_scene_folders = sorted(
        viz_scene_folders, key=lambda x: float(re.findall("(\d+)", x)[0])
    )
    sort_nicely(viz_scene_folders)

    return (cm_scene_folders, viz_scene_folders)


def get_source_dir() -> str:
    current_dir = os.path.dirname(os.path.realpath(__file__))
    source_dir = os.path.dirname(os.path.dirname(current_dir))
    return source_dir


def get_asset_dir() -> str:
    source_dir = get_source_dir()
    assets_dir = os.path.join(source_dir, "assets")
    return assets_dir


def get_image_dir(config: Dict[str, Union[str, float, bool]]) -> str:
    assets_dir = get_asset_dir()
    img_dir = os.path.join(assets_dir, "images", f"{config.scene_name}")
    mkdir_not_exists(img_dir)

    return img_dir


def get_video_dir(config: Dict[str, Union[str, float, bool]]) -> str:
    assets_dir = get_asset_dir()
    video_dir = os.path.join(assets_dir, "videos")
    mkdir_not_exists(video_dir)
    return video_dir
