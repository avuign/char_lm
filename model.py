import torch
import torch.nn as nn


# Model
class CharModel(nn.Module):
    def __init__(self, context_size, embedding_dim):
        super().__init__()
        self.E = nn.Embedding(27, embedding_dim)
        self.fc1 = nn.Linear(context_size * embedding_dim, 128)
        self.fc2 = nn.Linear(128, 27)

    def forward(self, x):
        x = self.E(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
