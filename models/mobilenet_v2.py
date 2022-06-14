import math
import torch
from torch import nn


class SiLU(nn.Module):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    '''

    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def ConvSiLU(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    return nn.Sequential(
           nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
           nn.BatchNorm2d(num_features=out_channels),
           SiLU())


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, reduction=4):  # r in squeeze-and-excitation optimization
        super(InvertedResidual, self).__init__()
        self.use_residual = (in_channels == out_channels) and (stride == 1)
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = math.floor(in_channels / reduction)

        if self.expand:
            self.expand_conv = ConvSiLU(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0)

        self.conv = nn.Sequential(
            ConvSiLU(in_channels=hidden_dim, out_channels=hidden_dim,
                     kernel_size=3, stride=stride, padding=1, groups=hidden_dim),       # deepwise
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, bias=False),    # pointwise
            nn.BatchNorm2d(num_features=out_channels))

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs
        if self.use_residual:
            return self.conv(x) + inputs
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = 32
        layers = [ConvSiLU(in_channels=3, out_channels=input_channel, kernel_size=3, stride=2, padding=1)]

        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = c
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)

        # building last several layers
        output_channel = 1280
        self.conv = ConvSiLU(in_channels=input_channel, out_channels=output_channel, kernel_size=1, stride=1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()   