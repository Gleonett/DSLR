import torch


def get_device(device):
    if "cpu" not in device:
        if torch.cuda.is_available():
            device = torch.device(device)
        else:
            exit("Cuda not available")
    else:
        device = torch.device(device)
    return device


def to_tensor(x, device, dtype):
    return torch.from_numpy(x).to(device, dtype)
