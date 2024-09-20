import pytest
import torch

from funlib.learn.torch.models import ResNet2D, ResNet3D


@pytest.mark.parametrize("version", [18, 34])
def test_resnet2d(version):
    net = ResNet2D(10, input_channels=2, start_channels=12, version=version)
    assert net(torch.rand(1, 2, 64, 64)).shape == (1, 10)
    # Try with a different shape, should still work
    assert net(torch.rand(1, 2, 32, 32)).shape == (1, 10)


@pytest.mark.parametrize("version", [18, 34])
def test_resnet3d(version):
    net = ResNet3D(10, input_channels=2, start_channels=12, version=version)
    assert net(torch.rand(1, 2, 64, 64, 64)).shape == (1, 10)
    # Try with a different shape, should still work
    assert net(torch.rand(1, 2, 32, 32, 32)).shape == (1, 10)
