"""
2D ResNet implementation based on code from https://github.com/funkelab/dapi/blob/main/dapi_networks/ResNet.py
"""

import torch
from torch import nn


class ResNet(nn.Module):

    def __init__(
        self,
        output_classes,
        input_channels=1,
        start_channels=12,
        version=18,
        dimension=2,
    ):
        """
        Args:
            output_classes: Number of output classes

            input_size: Size of input images

            input_channels: Number of input channels

            start_channels: Number of channels in first convolutional layer

            version: ResNet version (18, 34), defaults to 18

            dimension: Dimension of the input images (2, 3), defaults to 2
        """
        super(ResNet, self).__init__()
        self.dimension = dimension
        if self.dimension == 2:
            self.conv_layer = nn.Conv2d
            self.bn_layer = nn.BatchNorm2d
            self.avgpool_layer = nn.AdaptiveAvgPool2d
        elif self.dimension == 3:
            self.conv_layer = nn.Conv3d
            self.bn_layer = nn.BatchNorm3d
            self.avgpool_layer = nn.AdaptiveAvgPool3d
        self.in_channels = start_channels
        self.conv = self.conv_layer(
            input_channels,
            self.in_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
        )
        self.bn = self.bn_layer(self.in_channels)
        self.relu = nn.ReLU()

        current_channels = self.in_channels
        self.layers = nn.ModuleList()

        # Define number of blocks for each layer
        if version == 18:
            blocks = [2, 2, 2, 2]
        elif version == 34:
            blocks = [3, 4, 6, 3]

        for i, block in enumerate(blocks):
            self.layers.append(
                self.make_layer(ResidualBlock, current_channels, block, 2)
            )
            if i != 3:
                current_channels *= 2

        self.avgpool = self.avgpool_layer((1,) * self.dimension)
        self.fc = nn.Linear(current_channels, output_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or self.in_channels != out_channels:
            downsample = nn.Sequential(
                self.conv_layer(
                    self.in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=stride,
                    bias=True,
                ),
                self.bn_layer(out_channels),
            )
        layers = []
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride,
                downsample,
                dimension=self.dimension,
            )
        )
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, dimension=self.dimension))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        for layer in self.layers:
            out = layer(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# Residual block
class ResidualBlock(nn.Module):

    def __init__(
        self, in_channels, out_channels, stride=1, downsample=None, dimension=2
    ):
        super(ResidualBlock, self).__init__()
        self.dimension = dimension
        if self.dimension == 2:
            self.conv_layer = nn.Conv2d
            self.bn_layer = nn.BatchNorm2d
        elif self.dimension == 3:
            self.conv_layer = nn.Conv3d
            self.bn_layer = nn.BatchNorm3d
        # Biases are handled by BN layers
        self.conv1 = self.conv_layer(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=True,
        )
        self.bn1 = self.bn_layer(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = self.conv_layer(
            out_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn2 = self.bn_layer(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = nn.ReLU()(out)
        return out


# Convenience classes
class ResNet2D(ResNet):
    def __init__(self, output_classes, input_channels=1, start_channels=12, version=18):
        super().__init__(
            output_classes, input_channels, start_channels, version, dimension=2
        )


class ResNet3D(ResNet):
    def __init__(self, output_classes, input_channels=1, start_channels=12, version=18):
        super().__init__(
            output_classes, input_channels, start_channels, version, dimension=3
        )
        assert self.dimension == 3
