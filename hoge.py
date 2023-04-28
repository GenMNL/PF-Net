import torch
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [4, 6, 6]])
b = np.array([[1, 2, 3], [7, 8, 9], [4, 5, 6]])

c = np.concatenate([a, b], axis=0)
print(c)

without_unique, count = np.unique(c, axis=0, return_counts=True)

mask = count > 1
print(mask)
diff = without_unique[mask]
# diff = without_unique[mask, :]
print(diff)
