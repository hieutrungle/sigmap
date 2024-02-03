import yaml
import argparse
import os


def main():
    args = create_argparser().parse_args()
    with open(args.config_file) as istream:
        ymldoc = yaml.safe_load(istream)
        ymldoc["rx_position"] = args.rx_position

    # add tmp to args.config_file
    tmp_file = args.config_file.split(".")[0] + "_tmp.yaml"
    with open(tmp_file, "w") as ostream:
        yaml.dump(ymldoc, ostream, default_flow_style=False, sort_keys=False)

    # move/overwrite tmp file to output file
    os.rename(tmp_file, args.output)


def create_argparser() -> argparse.ArgumentParser:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument(
        "--rx_position",
        "-rx_pos",
        nargs="*",  # 0 or more values expected => creates a list
        type=float,
        default=[0, 0, 0],  # default if nothing is provided
        required=True,
    )
    return parser


if __name__ == "__main__":
    main()
