import torch

a = torch.randn(3, 3)
b = torch.tensor([[2, 1], [0, 1]])
print(b)
print(a)
# print(a[[0, 1], [[0, 1], [0, 1]]])
print(a[b, b])
