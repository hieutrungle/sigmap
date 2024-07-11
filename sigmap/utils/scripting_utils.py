import os
from typing import Dict, List, Union, Tuple
import argparse
import re

import time

import sigmap.drl.env_configs
from sigmap.drl.infrastructure.logger import TensorboardLogger
import argparse
from sigmap.utils import utils


class Config:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return str(self.__dict__)


def make_sionna_config(config_file: str) -> Config:
    config = Config()
    config_kwargs = utils.load_yaml_file(config_file)
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


def make_drl_config(config_file: str, args: argparse.Namespace) -> Config:
    config = Config()
    config_kwargs = utils.load_yaml_file(config_file)
    base_config_name = config_kwargs.pop("base_config")
    config_kwargs.update({"args": args})
    config.__dict__.update(
        sigmap.drl.env_configs.configs[base_config_name](**config_kwargs)
    )
    return config


def make_tensorboard_logger(config: dict) -> TensorboardLogger:
    logdir = os.path.join(config.saved_path, time.strftime("%d-%m-%Y_%H-%M-%S"))

    return TensorboardLogger(logdir)


def add_dict_to_argparser(
    parser: argparse.ArgumentParser,
    default_dict: Dict[str, Union[str, float, bool]],
) -> None:
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(
    args: argparse.Namespace,
    keys: List[str],
) -> Dict[str, Union[str, float, bool]]:
    return {k: getattr(args, k) for k in keys}


def str2bool(v: str) -> bool:
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
