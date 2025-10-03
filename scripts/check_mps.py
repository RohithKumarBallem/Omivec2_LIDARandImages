import torch
print("PyTorch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    x = torch.ones(2, device="mps")
    print("Tensor on MPS:", x)
else:
    print("Falling back to CPU.")
