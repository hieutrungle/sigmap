import bpy
from mathutils import Vector
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


def get_bisector_pt(tile_center: Vector, tx_pos: Vector, rx_pos: Vector) -> Vector:
    """
    Get the bisector point of the line between tx and rx.

    tile_center: A
    tx_pos: B
    rx_pos: C
    bisector_pt: D

    ratio = -|AB|/|AC|

    vec(BD) = ratio * vec(CD)

    x_D = 1/(1 - ratio) * (x_B - ratio * x_C) \n
    y_D = 1/(1 - ratio) * (y_B - ratio * y_C) \n
    z_D = 1/(1 - ratio) * (z_B - ratio * z_C)
    """
    tile_tx_len = (tx_pos - tile_center).length  # |AB|
    tile_rx_len = (rx_pos - tile_center).length  # |AC|
    ratio = -tile_tx_len / tile_rx_len
    bisector_pt = 1 / (1 - ratio) * (tx_pos - ratio * rx_pos)
    return bisector_pt


def get_midpoint(pt1, pt2):
    x = (pt1[0] + pt2[0]) / 2
    y = (pt1[1] + pt2[1]) / 2
    z = (pt1[2] + pt2[2]) / 2
    return [x, y, z]


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
    midpoint = get_bisector_pt(Vector(tile_center), Vector(tx_pos), Vector(rx_pos))
    return compute_rot_angle(tile_center, midpoint)


def compute_rot_angle(
    tile_center: list,
    pt: list,
) -> Tuple[float, float, float]:
    """Compute the rotation angles for the tile.
    return: (r, theta, phi)
        `r`: distance from the tile center to a point
        `theta`: rotation in y-axis
        `phi`: rotation in z-axis
    """
    x = tile_center[0] - pt[0]
    y = tile_center[1] - pt[1]
    z = tile_center[2] - pt[2]

    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.acos(z / r)  # rotation in y-axis
    phi = sign(y) * math.acos(x / math.sqrt(x**2 + y**2))  # rotation in z-axis

    return (r, theta, phi)
