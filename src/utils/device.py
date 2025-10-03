import os
import torch

def get_device():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # Optional: relax MPS watermark to reduce OOM (use with care)
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
