import torch
import logging

logger = logging.getLogger(__name__)


class Vgg2D(torch.nn.Module):
    def __init__(
            self,
            input_size,
            fmaps=32,
            downsample_factors=[(2, 2), (2, 2), (2, 2), (2, 2)],
            output_classes=6,
            input_fmaps=1):

        super(Vgg2D, self).__init__()

        current_fmaps = input_fmaps
        current_size = tuple(input_size)

        features = []
        for i in range(len(downsample_factors)):

            features += [
                torch.nn.Conv2d(
                    current_fmaps,
                    fmaps,
                    kernel_size=3,
                    padding=1),
                torch.nn.BatchNorm2d(fmaps),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(
                    fmaps,
                    fmaps,
                    kernel_size=3,
                    padding=1),
                torch.nn.BatchNorm2d(fmaps),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(downsample_factors[i])
            ]

            current_fmaps = fmaps
            fmaps *= 2

            size = tuple(
                int(c/d)
                for c, d in zip(current_size, downsample_factors[i]))
            check = (
                s*d == c
                for s, d, c in zip(size, downsample_factors[i], current_size))
            assert all(check), \
                "Can not downsample %s by chosen downsample factor" % \
                (current_size,)
            current_size = size

            logger.info(
                "VGG level %d: (%s), %d fmaps",
                i,
                current_size,
                current_fmaps)

        self.features = torch.nn.Sequential(*features)

        classifier = [
            torch.nn.Linear(
                current_size[0] *
                current_size[1] *
                current_fmaps,
                4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(
                4096,
                4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(
                4096,
                output_classes)
        ]

        self.classifier = torch.nn.Sequential(*classifier)

    def forward(self, raw):

        f = self.features(raw)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)

        return y
