import torch
import torch.nn as nn

# tensors are numpy arrays
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(a + b)
print(a @ b)

# gradient
x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2 * x + 1
y.backward()
print(x.grad)


# dense layer
layer = nn.Linear(784, 128)  # same as Dense(128) with 784 inputs
print(layer.weight.shape)
print(layer.bias.shape)


# Model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
