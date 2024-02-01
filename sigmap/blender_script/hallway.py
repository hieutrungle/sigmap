import bpy
from mathutils import Vector
import math
import os
import os, sys, inspect

# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(
    os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])
)
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

import bl_utils, bl_parser


def export_beamfocusing_simple_hallway(args, config):
    # Collection
    tiles = bpy.data.collections["Reflector"].objects

    for frame in range(1, 2):
        for tile in tiles:
            local_bbox_center = 0.125 * sum(
                (Vector(b) for b in tile.bound_box), Vector()
            )
            global_bbox_center = tile.matrix_world @ local_bbox_center

            # Compute rotation angles
            r, theta, phi = bl_utils.compute_rot_angle_txrx(
                global_bbox_center, config.tx_position, config.rx_position
            )

            tile.rotation_euler = [0, theta, phi]
            tile.scale = [0.1, 0.1, 0.02]

        # Saving to mitsuba format for Sionna
        print(
            f"\nsaving with index: {args.index},  rx_pos: {config.rx_position} and tx_pos: {config.tx_position}"
        )

        # Save files without ceiling
        folder_dir = os.path.join(
            args.output_dir,
            f"{config.scene_name}",
            f"idx_{args.index}_rx_pos_{config.rx_position}",
        )
        bl_utils.mkdir_with_replacement(folder_dir)
        bl_utils.save_mitsuba_xml(
            folder_dir, config.mitsuba_filename, ["Reflector", "Wall", "Floor"]
        )

        # Save files with ceiling
        folder_dir = os.path.join(
            args.output_dir,
            f"{config.scene_name}",
            f"ceiling_idx_{args.index}_rx_pos_{config.rx_position}",
        )
        bl_utils.mkdir_with_replacement(folder_dir)
        bl_utils.save_mitsuba_xml(
            folder_dir,
            config.mitsuba_filename,
            ["Reflector", "Wall", "Floor", "Ceiling"],
        )


def main():
    args = create_argparser().parse_args()
    config = bl_utils.make_conf(args.config_file)

    if config.scene_name == "beamfocusing_simple_hallway":
        export_beamfocusing_simple_hallway(args, config)
    else:
        raise ValueError(f"Unknown scene name: {config.scene_name}")


def create_argparser() -> bl_parser.ArgumentParserForBlender:
    """Parses command line arguments."""
    parser = bl_parser.ArgumentParserForBlender()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--index", type=int, default=0)
    return parser


if __name__ == "__main__":
    main()
