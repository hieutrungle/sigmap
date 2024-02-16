"""
Path: sigmap/drl/infrastructure/pytorch_utils.py
"""

from typing import Union, List
import torch
import torch.nn as nn
import numpy as np
from sigmap.utils import logger

Activation = Union[str, nn.Module]

_str_to_activation = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "softplus": nn.Softplus(),
    "identity": nn.Identity(),
}

DEVICE = None


def build_mlp(
    input_size: int,
    output_size: int,
    hidden_sizes: List[int],
    # n_layers: int,
    activation: Activation = "relu",
    output_activation: Activation = "identity",
):
    """
    Builds a feedforward neural network

    arguments:
        input_size: size of the input layer
        output_size: size of the output layer
        hidden_size: dimension of each hidden layer
        n_layers: number of hidden layers
        activation: activation of each hidden layer
        output_activation: activation of the output layer

    returns:
        output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    layers = []
    in_size = input_size
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(in_size, hidden_size))
        layers.append(activation)
        in_size = hidden_size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)

    mlp = nn.Sequential(*layers)
    mlp.to(DEVICE)
    return mlp


def init_gpu(use_gpu: bool = True, gpu_id: int = 0):
    global DEVICE
    if use_gpu and torch.cuda.is_available():
        DEVICE = torch.device(f"cuda:{gpu_id}")
        logger.log(f"Using GPU {gpu_id} for PyTorch")

    else:
        DEVICE = torch.device("cpu")
        logger.log("Using CPU for PyTorch")


def set_device(gpu_id: int):
    torch.cuda.set_device(gpu_id)


def from_numpy(data: Union[np.ndarray, dict], **kwargs):
    if isinstance(data, dict):
        return {k: from_numpy(v) for k, v in data.items()}
    else:
        data = torch.from_numpy(data, **kwargs)
        if data.dtype == torch.float64:
            data = data.float()
        return data.to(DEVICE)


def to_numpy(tensor: Union[torch.Tensor, dict]):
    if isinstance(tensor, dict):
        return {k: to_numpy(v) for k, v in tensor.items()}
    else:
        return tensor.to("cpu").detach().numpy()
