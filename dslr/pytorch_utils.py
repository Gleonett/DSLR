"""
Some functions for PyTorch
"""

import torch
import numpy as np


def get_device(device: str) -> torch.device:
    """
    :param device: "cpu", "cuda" or "cuda:{device_index}"
    :return: torch.device
    """
    if "cpu" not in device:
        if torch.cuda.is_available():
            device = torch.device(device)
        else:
            exit("Cuda not available")
    else:
        device = torch.device(device)
    return device


def to_tensor(x: np.ndarray,
              device: torch.device,
              dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(x).to(device, dtype)
