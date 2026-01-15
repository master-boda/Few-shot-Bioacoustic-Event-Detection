"""baseline model definitions."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    # 3x3 conv with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    # baseline resnet block

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        drop_rate: float = 0.0,
        drop_block: bool = False,
        block_size: int = 1,
    ):
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
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size

    def forward(self, x):
        self.num_batches_tracked += 1
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

        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        return out


class ResNet(nn.Module):
    # 9-layer resnet used in the baseline

    def __init__(
        self,
        block=BasicBlock,
        keep_prob: float = 1.0,
        avg_pool: bool = True,
        drop_rate: float = 0.1,
        dropblock_size: int = 5,
        pool_time_only: bool = False,
    ):
        self.inplanes = 1
        super().__init__()

        pool_stride = (2, 1) if pool_time_only else 2
        self.layer1 = self._make_layer(block, 64, stride=pool_stride, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 128, stride=pool_stride, drop_rate=drop_rate)
        self.layer3 = self._make_layer(
            block,
            64,
            stride=pool_stride,
            drop_rate=drop_rate,
            drop_block=True,
            block_size=dropblock_size,
        )
        self.layer4 = self._make_layer(
            block,
            64,
            stride=pool_stride,
            drop_rate=drop_rate,
            drop_block=True,
            block_size=dropblock_size,
        )
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        self.pool = nn.AdaptiveAvgPool2d((4, 2))
        self.embedding_dim = 64 * 4 * 2

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate: float = 0.0, drop_block: bool = False, block_size: int = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        (num_samples, seq_len, mel_bins) = x.shape
        x = x.view(-1, 1, seq_len, mel_bins)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # layer4 is disabled in the baseline
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x


def build_encoder(conf) -> nn.Module:
    # select encoder by config
    name = str(conf.train.get("encoder", "Resnet")).lower()
    use_delta_mfcc = bool(conf.features.get("use_delta_mfcc", False))
    n_mfcc = int(conf.features.get("n_mfcc", 0)) if use_delta_mfcc else 0
    feat_dim = int(conf.features.n_mels) + n_mfcc
    if name not in {"resnet"}:
        raise ValueError(f"unsupported encoder '{conf.train.get('encoder')}', only 'Resnet' is supported")
    model_cfg = conf.get("model", {})
    r_cfg = model_cfg.get("resnet", {})
    return ResNet(
        avg_pool=bool(r_cfg.get("avg_pool", True)),
        drop_rate=float(r_cfg.get("drop_rate", 0.1)),
        dropblock_size=int(r_cfg.get("dropblock_size", 5)),
        pool_time_only=bool(r_cfg.get("pool_time_only", False)),
    )
