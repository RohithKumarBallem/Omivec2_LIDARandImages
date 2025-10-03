import torch
from src.utils.device import get_device

dev = get_device()
print("Device:", dev)
x = torch.randn(2, 3, 64, 64, device=dev)
w = torch.randn(4, 3, 3, 3, device=dev)
y = torch.nn.functional.conv2d(x, w, padding=1)
print("OK, conv2d:", y.shape)
