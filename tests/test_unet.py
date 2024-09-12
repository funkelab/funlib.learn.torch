import warnings

import numpy as np
import pytest
import torch

from funlib.learn.torch import models

warnings.filterwarnings("error")


def test_creation():
    unet = models.UNet(
        in_channels=1,
        num_fmaps=3,
        fmap_inc_factor=2,
        downsample_factors=[[2, 2, 2], [2, 2, 2]],
    )

    x = np.zeros((1, 1, 100, 80, 48), dtype=np.float64)
    x = torch.from_numpy(x).float()

    y = unet.forward(x).data.numpy()

    assert y.shape == (1, 3, 60, 40, 8)

    unet = models.UNet(
        in_channels=1,
        num_fmaps=3,
        fmap_inc_factor=2,
        downsample_factors=[[2, 2, 2], [2, 2, 2]],
        num_fmaps_out=5,
    )

    y = unet.forward(x).data.numpy()

    assert y.shape == (1, 5, 60, 40, 8)


def test_shape_warning():
    x = np.zeros((1, 1, 100, 80, 48), dtype=np.float64)
    x = torch.from_numpy(x).float()

    # Should raise warning
    with pytest.raises(RuntimeError):
        unet = models.UNet(
            in_channels=1,
            num_fmaps=3,
            fmap_inc_factor=2,
            downsample_factors=[[2, 3, 2], [2, 2, 2]],
            num_fmaps_out=5,
        )
        unet.forward(x).data.numpy()


def test_multi_head():
    unet = models.UNet(
        in_channels=1,
        num_fmaps=3,
        fmap_inc_factor=2,
        downsample_factors=[[2, 2, 2], [2, 2, 2]],
        num_heads=3,
    )

    x = np.zeros((1, 1, 100, 80, 48), dtype=np.float64)
    x = torch.from_numpy(x).float()

    y = unet.forward(x)

    assert len(y) == 3
    assert y[0].data.numpy().shape == (1, 3, 60, 40, 8)
    assert y[1].data.numpy().shape == (1, 3, 60, 40, 8)
    assert y[2].data.numpy().shape == (1, 3, 60, 40, 8)
