from funlib.learn.torch import models
import numpy as np
import pytest
import torch
import unittest
import warnings
warnings.filterwarnings("error")

class TestVgg3D(unittest.TestCase):
    def test_creation(self):
        vgg = models.Vgg3D(
            input_size=(128,128,128))

        x = np.zeros((1,128,128,128), dtype=np.float64)
        x = torch.from_numpy(x).float()

        y = vgg.forward(x).data.numpy()
        print(y.shape)

        assert y.shape == (1, 6)


        vgg = models.Vgg3D(
            input_size=(128,128,128),
            fmap_inc=(4,1,1,1))

        x = np.zeros((1,128,128,128), dtype=np.float64)
        x = torch.from_numpy(x).float()

        y = vgg.forward(x).data.numpy()
        print(y.shape)

        assert y.shape == (1, 6)

        vgg = models.Vgg3D(
            input_size=(128,128,128),
            fmap_inc=(4,1,1,1),
            n_convolutions=(4,2,2,2))

        x = np.zeros((1,128,128,128), dtype=np.float64)
        x = torch.from_numpy(x).float()

        y = vgg.forward(x).data.numpy()
        print(y.shape)

        assert y.shape == (1, 6)


if __name__ == "__main__":
    unittest.main()
