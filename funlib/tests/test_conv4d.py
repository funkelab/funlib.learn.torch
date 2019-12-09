from funlib.learn.torch.models import Conv4d
import numpy as np
import torch
import unittest


class TestConv4D(unittest.TestCase):

    def test_conv4d(self):

        # Generate random input 4D tensor (+ batch dimension, + channel
        # dimension)
        np.random.seed(42)
        input_numpy = np.round(np.random.random((1, 1, 10, 11, 12, 13)) * 100)
        input_torch = torch.from_numpy(input_numpy).float()

        # Convolve with a randomly initialized kernel

        # Initialize the 4D convolutional layer with random kernels
        conv4d_layer = \
            Conv4d(
                in_channels=1,
                out_channels=1,
                kernel_size=(3, 3, 3, 3),
                bias_initializer=lambda x: torch.nn.init.constant_(x, 0))

        # Pass the input tensor through that layer
        output = conv4d_layer.forward(input_torch).data.numpy()

        # Select the 3D kernels for the manual computation and comparison
        kernels = [
            conv4d_layer.conv3d_layers[i].weight.data.numpy().flatten()
            for i in range(3)
        ]

        # Compare the conv4d_layer result and the manual convolution
        # computation at 3 randomly chosen locations
        for i in range(3):

            # Randomly choose a location and select the conv4d_layer output
            loc = [
                np.random.randint(0, output.shape[2] - 2),
                np.random.randint(0, output.shape[3] - 2),
                np.random.randint(0, output.shape[4] - 2),
                np.random.randint(0, output.shape[5] - 2)
            ]
            conv4d = output[0, 0, loc[0], loc[1], loc[2], loc[3]]

            # Select slices from the input tensor and compute manual
            # convolution
            slices = [
                input_numpy[
                    0, 0, loc[0] + j, loc[1]:loc[1] + 3,
                    loc[2]:loc[2] + 3, loc[3]:loc[3] + 3].flatten()
                for j in range(3)
            ]
            manual = np.sum([slices[j] * kernels[j] for j in range(3)])

            np.testing.assert_array_almost_equal(conv4d, manual, 3)

        # Convolve with a kernel initialized to be all ones
        conv4d_layer = \
            Conv4d(
                in_channels=1,
                out_channels=1,
                kernel_size=(3, 3, 3, 3),
                padding=1,
                kernel_initializer=lambda x: torch.nn.init.constant_(x, 1),
                bias_initializer=lambda x: torch.nn.init.constant_(x, 0))

        output = conv4d_layer.forward(input_torch).data.numpy()

        # Define relu(x) = max(x, 0) for simplified indexing below
        def relu(x: float) -> float:
            return x * (x > 0)

        # Compare the conv4d_layer result and the manual convolution
        # computation at 3 randomly chosen locations
        for i in range(3):

            # Randomly choose a location and select the conv4d_layer
            # output
            loc = [np.random.randint(0, output.shape[2] - 2),
                   np.random.randint(0, output.shape[3] - 2),
                   np.random.randint(0, output.shape[4] - 2),
                   np.random.randint(0, output.shape[5] - 2)]
            conv4d = output[0, 0, loc[0], loc[1], loc[2], loc[3]]

            # For a kernel that is all 1s, we only need to sum up the elements
            # of the input (the ReLU takes care of the padding!)
            manual = input_numpy[0, 0,
                                 relu(loc[0] - 1):loc[0] + 2,
                                 relu(loc[1] - 1):loc[1] + 2,
                                 relu(loc[2] - 1):loc[2] + 2,
                                 relu(loc[3] - 1):loc[3] + 2].sum()

            np.testing.assert_array_almost_equal(conv4d, manual, 3)
