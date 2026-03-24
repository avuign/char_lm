import torch

# tensors are numpy arrays
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(a + b)
print(a @ b)
