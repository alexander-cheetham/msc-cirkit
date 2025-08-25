from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch

ds = datasets.MNIST("./data", train=False, download=True, transform=transforms.PILToTensor())
loader = DataLoader(ds, batch_size=len(ds), shuffle=False)

X_u8, y = next(iter(loader))              # (N,1,28,28) uint8
X = X_u8.view(X_u8.size(0), -1).to(torch.int64)  # (N,784) int64 in [0,255]
torch.save((X, y), "mnist_flat_int64_cache.pt")
