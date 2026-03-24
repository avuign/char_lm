import torch
import torch.nn as nn

from config import HIDDEN_DIM


# Model
class CharModel(nn.Module):
    def __init__(self, context_size, embedding_dim, hidden_dims):
        super().__init__()
        self.E = nn.Embedding(27, embedding_dim)

        self.layers = nn.ModuleList()
        layer_sizes = (
            [context_size * embedding_dim] + hidden_dims + [27]
        )  # [27,hidden_dim[0], hidden_dim[1],...,27]
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        x = self.E(x)
        x = x.view(x.shape[0], -1)

        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.relu(x)
        x = self.layers[-1](x)
        return x
