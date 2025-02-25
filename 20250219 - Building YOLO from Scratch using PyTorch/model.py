import torch
import torch.nn as nn

darknet_config = [
    # (kernel size, filters, stride, padding)
    # M for max pooling
    (7, 64, 2, 3),
    'M',
    (3, 192, 1, 1),
    'M',

    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    'M',

    (1, 256, 1, 0),
    (3, 512, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),

    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    'M',

    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    (1, 512, 1, 0),
    (3, 1024, 1, 1),

    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        return x


class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(YOLOv1, self).__init__()
        self.backbone_config = darknet_config
        self.in_channels = in_channels
        self.darknet = self._create_backbone(self.backbone_config)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_backbone(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == str and x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
                
            else:
                layers += [CNNBlock(in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3])]
                in_channels = x[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (B * 5 + C)),
        )