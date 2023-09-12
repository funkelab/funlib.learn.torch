from funlib.learn.torch.models import ResNet2D, ResNet3D
import torch


def test_resnet2d():
    net = ResNet2D(10, input_channels=2, start_channels=12)
    assert net(torch.rand(1, 2, 64, 64)).shape == (1, 10)
    # Try with a different shape, should still work
    assert net(torch.rand(1, 2, 32, 32)).shape == (1, 10)


def test_resnet3d():
    net = ResNet3D(10, input_channels=2, start_channels=12)
    assert net(torch.rand(1, 2, 64, 64, 64)).shape == (1, 10)
    # Try with a different shape, should still work
    assert net(torch.rand(1, 2, 32, 32, 32)).shape == (1, 10)