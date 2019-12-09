# -*- coding: UTF-8 -*-
import torch


class Conv4d(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            padding_mode='zeros',
            dilation=1,
            groups=1,
            bias=True,
            bias_initializer=None,
            kernel_initializer=None):
        '''
        Performs a 4D convolution of the ``(t, z, y, x)`` dimensions of a
        tensor with shape ``(b, c, l, d, h, w)`` with ``k`` filters. The output
        tensor will be of shape ``(b, k, l', d', h', w')``. ``(l', d', h',
        w')`` will be smaller than ``(l, d, h, w)`` if a padding smaller than
        half of the kernel size was chosen.

        Args:

            in_channels (int):

                Number of channels in the input image.

            out_channels (int):

                Number of channels produced by the convolution.

            kernel_size (int or tuple):

                Size of the convolving kernel.

            stride (int or tuple, optional):

                Stride of the convolution. Default: 1

            padding (int or tuple, optional):

                Zero-padding added to all four sides of the input. Default: 0

            padding_mode (string, optional).

                Accepted values `zeros` and `circular`. Default: `zeros`

            dilation (int or tuple, optional):

                Spacing between kernel elements. Default: 1

            groups (int, optional):

                Number of blocked connections from input channels to output
                channels. Default: 1

            bias (bool, optional):

                If ``True``, adds a learnable bias to the output. Default:
                ``True``

            bias_initializer, kernel_initializer (callable):

                An optional initializer for the bias and the kernel weights.

        This operator realizes a 4D convolution by performing several 3D
        convolutions. The following example demonstrates how this works for a
        2D convolution as a sequence of 1D convolutions::

            I.shape == (h, w)
            k.shape == (U, V) and U%2 = V%2 = 1

            # we assume kernel is indexed as follows:
            u in [-U/2,...,U/2]
            v in [-V/2,...,V/2]

            (k*I)[i,j] = Σ_u Σ_v k[u,v] I[i+u,j+v]
                       = Σ_u (k[u]*I[i+u])[j]
            (k*I)[i]   = Σ_u k[u]*I[i+u]
            (k*I)      = Σ_u k[u]*I_u, with I_u[i] = I[i+u] shifted I by u

            Example:

                I = [
                    [0,0,0],
                    [1,1,1],
                    [1,1,0],
                    [1,0,0],
                    [0,0,1]
                ]

                k = [
                    [1,1,1],
                    [1,2,1],
                    [1,1,3]
                ]

                # convolve every row in I with every row in k, comments show
                # output row the convolution contributes to
                (I*k[0]) = [
                    [0,0,0], # I[0] with k[0] ⇒ (k*I)[ 1] ✔
                    [2,3,2], # I[1] with k[0] ⇒ (k*I)[ 2] ✔
                    [2,2,1], # I[2] with k[0] ⇒ (k*I)[ 3] ✔
                    [1,1,0], # I[3] with k[0] ⇒ (k*I)[ 4] ✔
                    [0,1,1]  # I[4] with k[0] ⇒ (k*I)[ 5]
                ]
                (I*k[1]) = [
                    [0,0,0], # I[0] with k[1] ⇒ (k*I)[ 0] ✔
                    [3,4,3], # I[1] with k[1] ⇒ (k*I)[ 1] ✔
                    [3,3,1], # I[2] with k[1] ⇒ (k*I)[ 2] ✔
                    [2,1,0], # I[3] with k[1] ⇒ (k*I)[ 3] ✔
                    [0,1,2]  # I[4] with k[1] ⇒ (k*I)[ 4] ✔
                ]
                (I*k[2]) = [
                    [0,0,0], # I[0] with k[2] ⇒ (k*I)[-1]
                    [4,5,2], # I[1] with k[2] ⇒ (k*I)[ 0] ✔
                    [4,2,1], # I[2] with k[2] ⇒ (k*I)[ 1] ✔
                    [1,1,0], # I[3] with k[2] ⇒ (k*I)[ 2] ✔
                    [0,3,1]  # I[4] with k[2] ⇒ (k*I)[ 3] ✔
                ]

                # the sum of all valid output rows gives k*I (here shown for
                # row 2)
                (k*I)[2] = (
                    [2,3,2] +
                    [3,3,1] +
                    [1,1,0] +
                ) = [6,7,3]
        '''

        super(Conv4d, self).__init__()

        # ---------------------------------------------------------------------
        # Assertions for constructor arguments
        # ---------------------------------------------------------------------

        assert len(kernel_size) == 4, \
            '4D kernel size expected!'
        assert stride == 1, \
            'Strides other than 1 not yet implemented!'
        assert dilation == 1, \
            'Dilation rate other than 1 not yet implemented!'
        assert groups == 1, \
            'Groups other than 1 not yet implemented!'

        # ---------------------------------------------------------------------
        # Store constructor arguments
        # ---------------------------------------------------------------------

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        self.bias = bias

        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer

        # ---------------------------------------------------------------------
        # Construct 3D convolutional layers
        # ---------------------------------------------------------------------

        # Shortcut for kernel dimensions
        (l_k, d_k, h_k, w_k) = self.kernel_size

        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv3d_layers = torch.nn.ModuleList()

        for i in range(l_k):

            # Initialize a Conv3D layer
            conv3d_layer = torch.nn.Conv3d(in_channels=self.in_channels,
                                           out_channels=self.out_channels,
                                           kernel_size=(d_k, h_k, w_k),
                                           padding=self.padding)

            # Apply initializer functions to weight and bias tensor
            if self.kernel_initializer is not None:
                self.kernel_initializer(conv3d_layer.weight)
            if self.bias_initializer is not None:
                self.bias_initializer(conv3d_layer.bias)

            # Store the layer
            self.conv3d_layers.append(conv3d_layer)

    # -------------------------------------------------------------------------

    def forward(self, input):

        # Define shortcut names for dimensions of input and kernel
        (b, c_i, l_i, d_i, h_i, w_i) = tuple(input.shape)
        (l_k, d_k, h_k, w_k) = self.kernel_size

        # Compute the size of the output tensor based on the zero padding
        l_o = l_i + 2 * self.padding - l_k + 1

        # Output tensors for each 3D frame
        frame_results = l_o * [None]

        # Convolve each kernel frame i with each input frame j
        for i in range(l_k):

            for j in range(l_i):

                # Add results to this output frame
                out_frame = j - (i - l_k // 2) - (l_i - l_o) // 2
                if out_frame < 0 or out_frame >= l_o:
                    continue

                frame_conv3d = \
                    self.conv3d_layers[i](input[:, :, j, :]
                                          .view(b, c_i, d_i, h_i, w_i))

                if frame_results[out_frame] is None:
                    frame_results[out_frame] = frame_conv3d
                else:
                    frame_results[out_frame] += frame_conv3d

        return torch.stack(frame_results, dim=2)
