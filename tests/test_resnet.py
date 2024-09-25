import pytest
import torch

from funlib.learn.torch.models import ResNet


@pytest.mark.parametrize("num_blocks", [[2, 2, 2, 2], [3, 4, 6, 3], [3, 4, 23, 3]])
@pytest.mark.parametrize("dimension", [2, 3])
def test_resnet(num_blocks, dimension):
    net = ResNet(
        10,
        input_channels=2,
        start_channels=12,
        num_blocks=num_blocks,
        dimension=dimension,
    )
    input_spatial_shape = (64,) * dimension
    assert net(torch.rand(1, 2, *input_spatial_shape)).shape == (1, 10)
    # Try with a different shape, should still work
    input_spatial_shape = (32,) * dimension
    assert net(torch.rand(1, 2, *input_spatial_shape)).shape == (1, 10)
