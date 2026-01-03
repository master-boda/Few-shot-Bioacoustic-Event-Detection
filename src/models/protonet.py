"""Minimal prototypical network backbone with optional IMT-style ResNet encoder."""

import torch
from torch import nn, Tensor


def _conv3x3(in_planes: int, out_planes: int, stride=1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class _IMTBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride=(2, 2), downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = _conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        out = self.maxpool(out)
        return out


class _IMTResNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.inplanes = in_channels
        self.layer1 = self._make_layer(_IMTBasicBlock, 64, stride=(2, 2))
        self.layer2 = self._make_layer(_IMTBasicBlock, 128, stride=(2, 2))
        self.layer3 = self._make_layer(_IMTBasicBlock, 256, stride=(1, 2))
        self.pool = nn.AdaptiveMaxPool2d((8, 1))
        self.output_dim = 256 * 8

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes: int, stride=(2, 2)) -> nn.Sequential:
        downsample = None
        if self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return x.flatten(1)


class ProtoNet(nn.Module):
    """Lightweight prototypical network for few-shot classification on log-mel inputs."""

    def __init__(
        self,
        input_channels: int = 1,
        hidden_size: int = 128,
        embedding_dim: int = 64,
        num_blocks: int = 1,
        channel_mult: int = 2,
        encoder_type: str = "simple_cnn",
    ):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == "imt_resnet":
            self.encoder = _IMTResNetEncoder(in_channels=input_channels)
            self.proj = nn.Linear(self.encoder.output_dim, embedding_dim)
        else:
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
        # x: (batch, channels, n_mels, time)
        feats = self.encoder(x)
        feats = feats.flatten(1)
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
