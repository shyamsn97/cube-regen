import torch


def preferred_device(device=None):
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if has_mps():
        return torch.device("mps")
    return torch.device("cpu")


def has_mps():
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())
