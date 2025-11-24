"""Minimal prototypical network backbone."""

import torch
from torch import nn, Tensor


class ProtoNet(nn.Module):
    """Lightweight prototypical network for few-shot classification."""

    def __init__(self, input_channels: int = 1, hidden_size: int = 128, embedding_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(embedding_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, channels, time)
        return self.encoder(x)

    @staticmethod
    def compute_prototypes(embeddings: Tensor, labels: Tensor, n_way: int) -> Tensor:
        """Compute class prototypes for support embeddings."""
        prototypes = []
        for c in range(n_way):
            mask = labels == c
            prototypes.append(embeddings[mask].mean(dim=0))
        return torch.stack(prototypes)

    @staticmethod
    def pairwise_distances(x: Tensor, y: Tensor) -> Tensor:
        """Compute squared Euclidean distances between two sets of vectors."""
        return ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(dim=2)

