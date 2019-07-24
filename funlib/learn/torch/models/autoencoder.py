import torch

class Autoencoder(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            downsampling_factors,
            fmaps,
            fmul,
            fmaps_bottle,
            kernel_size=3):

        super(Autoencoder, self).__init__()

        out_channels = in_channels

        encoder = []

        for downsampling_factor in downsampling_factors:

            encoder.append(
                    torch.nn.Conv3d(
                        in_channels,
                        fmaps,
                        kernel_size))
            encoder.append(
                    torch.nn.ReLU(inplace=True))
            encoder.append(
                    torch.nn.Conv3d(
                        fmaps,
                        fmaps,
                        kernel_size))
            encoder.append(
                    torch.nn.ReLU(inplace=True))
            encoder.append(
                    torch.nn.MaxPool3d(downsampling_factor))

            in_channels = fmaps

            fmaps = fmaps * fmul

        encoder.append(
            torch.nn.Conv3d(
                in_channels,
                fmap_bottle,
                kernel_size))
        encoder.append(
            torch.nn.ReLU(inplace=True))

        self.encoder = torch.nn.Sequential(*encoder)

        decoder = []

        fmaps = in_channels

        decoder.append(
            torch.nn.Conv3d(
                fmaps_bottle,
                fmaps,
                kernel_size))
        decoder.append(
            torch.nn.ReLU(inplace=True))

        for downsampling_factor in downsampling_factors[::-1]:

            fmaps = in_channels / fmul

            decoder.append(
                torch.nn.Upsample(
                    scale_factor=downsampling_factor,
                    mode='trilinear'))
            decoder.append(
                torch.nn.Conv3d(
                    in_channels,
                    fmaps,
                    kernel_size))
            decoder.append(
                torch.nn.ReLU(inplace=True))
            decoder.append(
                torch.nn.Conv3d(
                    fmaps,
                    fmaps,
                    kernel_size))
            decoder.append(
                torch.nn.ReLU(inplace=True))

            in_channels = fmaps

        decoder.append(
            torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size))

        self.decoder = torch.nn.Sequential(*decoder)


    def forward(self, x):

        enc = self.encoder(x)

        dec = self.decoder(enc)

        return dec



