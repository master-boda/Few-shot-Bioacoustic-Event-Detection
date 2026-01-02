"""Minimal prototypical network backbone."""

import torch
from torch import nn, Tensor


class ProtoNet(nn.Module):
    """Lightweight prototypical network for few-shot classification on log-mel inputs."""

    def __init__(
        self,
        input_channels: int = 1,
        hidden_size: int = 128,
        embedding_dim: int = 64,
        num_blocks: int = 1,
        channel_mult: int = 2,
    ):
        super().__init__()
        channels = [hidden_size]
        for _ in range(1, max(1, num_blocks)):
            channels.append(hidden_size * channel_mult)

        layers = []
        in_ch = input_channels
        for i, out_ch in enumerate(channels):
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            )
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2))
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool2d((1, embedding_dim)))  # pool freq to 1, time to embedding_dim
        self.encoder = nn.Sequential(*layers)
        self.proj = nn.Linear(in_ch * embedding_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, channels=1, n_mels, time)
        feats = self.encoder(x)  # (batch, hidden, 1, embedding_dim)
        feats = feats.flatten(1)  # (batch, hidden * embedding_dim)
        return self.proj(feats)   # (batch, embedding_dim)

    def classify(self, embeddings: Tensor) -> Tensor:
        return self.classifier(embeddings)

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
