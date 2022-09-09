import torch

a = torch.randn(3, 3)
b = torch.unsqueeze(a, dim=1)
a += 1
print(a)
print(b)
