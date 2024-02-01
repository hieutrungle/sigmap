import bpy
import os
import shutil
import math
from typing import Tuple
import re
import yaml


class Config:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return str(self.__dict__)


def make_conf(conf_file: str) -> Config:
    config = Config()
    config_kwargs = {}
    with open(conf_file, "r") as f:
        config_kwargs = yaml.safe_load(f)
    for k, v in config_kwargs.items():
        if isinstance(v, str):
            if v.lower() == "true":
                config_kwargs[k] = True
            elif v.lower() == "false":
                config_kwargs[k] = False
            elif v.isnumeric():
                config_kwargs[k] = float(v)
            elif re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$", v):
                config_kwargs[k] = float(v)

    config.__dict__.update(config_kwargs)
    return config


class Select(bpy.types.Operator):
    """Tooltip"""

    bl_idname = "outliner.simple_operator"
    bl_label = "Simple Outliner Operator"

    @classmethod
    def poll(cls, context):
        return context.area.type == "OUTLINER"

    def execute(self, context):
        sel = []
        for i in context.selected_ids:
            if i.bl_rna.identifier == "Collection":
                sel.append(i)

        for i in sel:
            bpy.ops.object.select_all(action="DESELECT")
            for o in i.objects:
                o.select_set(True)

        return {"FINISHED"}


def mkdir_not_exists(folder_dir):
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)


def mkdir_with_replacement(folder_dir):
    if os.path.exists(folder_dir):
        shutil.rmtree(folder_dir)
    os.makedirs(folder_dir)


def get_midpoint(pt1, pt2):
    x = (pt1[0] + pt2[0]) / 2
    y = (pt1[1] + pt2[1]) / 2
    z = (pt1[2] + pt2[2]) / 2
    return [x, y, z]


def sign(num):
    return -1 if num < 0 else 1


def select_collection(collections):
    bpy.ops.object.select_all(action="DESELECT")
    for collection in collections:
        for object in bpy.data.collections[collection].objects:
            object.select_set(True)


def save_mitsuba_xml(folder_dir, filename, collections):
    filepath = os.path.join(folder_dir, f"{filename}.xml")
    bpy.ops.object.select_all(action="DESELECT")
    select_collection(collections)
    bpy.ops.export_scene.mitsuba(
        filepath=filepath,
        check_existing=True,
        filter_glob="*.xml",
        use_selection=True,
        split_files=False,
        export_ids=True,
        ignore_background=True,
        axis_forward="Y",
        axis_up="Z",
    )


def compute_rot_angle_txrx(
    tile_center: list,
    tx_pos: list,
    rx_pos: list,
) -> Tuple[float, float, float]:
    """Compute the rotation angles for the tile.
    return: (r, theta, phi)
        `r`: distance from the tile center to the midpoint of tx and rx
        `theta`: rotation in y-axis
        `phi`: rotation in z-axis
    """
    midpoint = get_midpoint(tx_pos, rx_pos)
    return compute_rot_angle_midpt(tile_center, midpoint)


def compute_rot_angle_midpt(
    tile_center: list,
    midpoint: list,
) -> Tuple[float, float, float]:
    """Compute the rotation angles for the tile.
    return: (r, theta, phi)
        `r`: distance from the tile center to the midpoint of tx and rx
        `theta`: rotation in y-axis
        `phi`: rotation in z-axis
    """
    x = tile_center[0] - midpoint[0]
    y = tile_center[1] - midpoint[1]
    z = tile_center[2] - midpoint[2]

    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.acos(z / r)
    phi = sign(y) * math.acos(x / math.sqrt(x**2 + y**2))

    theta = math.degrees(theta)  # rotation in y-axis
    phi = math.degrees(phi)  # rotation in z-axis
    return (r, theta, phi)
