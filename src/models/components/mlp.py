from typing import Optional

import torch
from torch import nn

class TinyMLP(nn.Module):
    """Simple MLP backbone over flattened feature vectors."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """The last hidden layer, hidden_dim // 2 wide: what the output layer reads."""
        return self.net[:-1](x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net[-1](self.get_embeddings(x))
