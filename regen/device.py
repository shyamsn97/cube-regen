import torch


def preferred_device(device=None):
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if has_mps_conv3d():
        return torch.device("mps")
    return torch.device("cpu")


def has_mps():
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def has_mps_conv3d():
    if not has_mps():
        return False
    try:
        layer = torch.nn.Conv3d(1, 1, kernel_size=1).to("mps")
        sample = torch.zeros(1, 1, 2, 2, 2, device="mps")
        layer(sample)
    except RuntimeError:
        return False
    return True
