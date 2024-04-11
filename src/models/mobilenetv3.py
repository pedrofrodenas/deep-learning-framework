from typing import List
from functools import partial

import torch.nn as nn
from torch import Tensor
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation

from .utils import make_divisible

class InvertedResidual(nn.Module):
    def __init__(self,
                 inp,
                 kernel_size,
                 hidden,
                 out, 
                 SE,
                 activation,
                 stride,
                 norm_layer):
        
        super(InvertedResidual, self).__init__()

        self.stride = stride

        activation_layer = nn.Hardswish if activation=="HS" else nn.ReLU

        # If input channels are equal to input channels we can
        # use residual connection
        self.use_res_connect = self.stride == 1 and inp == out

        layers: List[nn.Module] = []

        # Channel Expansion Layer
        if hidden != inp:
            layers.append(
                Conv2dNormActivation(
                    inp,
                    hidden,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # Depth-Wise Convolution 3x3 kernel
        layers.append(
            Conv2dNormActivation(
                hidden,
                hidden,
                kernel_size=kernel_size,
                stride=stride,
                groups=hidden,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )
        # Squeeze-and-Excite, if its present in this layer
        if SE:
            squeeze_channels = make_divisible(hidden // 4, 8)
            layers.append(SqueezeExcitation(hidden, squeeze_channels, activation=nn.Hardsigmoid))

        # Linear projection, (no activation function)
        layers.append(
            Conv2dNormActivation(hidden, out, kernel_size=1, norm_layer=norm_layer, 
                                 activation_layer=None)
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class Mobilenetv3(nn.Module):
    def __init__(self,
                 num_classes: int = 1000,
                 width_mult: float = 1.0,
                 dropout: float = 0.2,
                 pretrained = True,
                 **kwargs):
        super(Mobilenetv3, self).__init__()

        first_layer_output = 16
        last_channel = 1024
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        settings = [
            # inputs, k_size, expansion_dim, outputs, Squeeze-Excite, Activation, Stride, 
            [3, 16,  16, True,  "RE", 2],  # C1
            [3, 72,  24, False, "RE", 2],  # C2
            [3, 88,  24, False, "RE", 1],
            [5, 96,  40, True,  "HS", 2],  # C3
            [5, 240, 40, True,  "HS", 1],
            [5, 240, 40, True,  "HS", 1],
            [5, 120, 48, True,  "HS", 1],
            [5, 144, 48, True,  "HS", 1],
            [5, 288, 96, True,  "HS", 2],  # C4
            [5, 576, 96, True,  "HS", 1],
            [5, 576, 96, True,  "HS", 1]
        ]

        features: List[nn.Module] = []

        # First layer, downsampling / 2
        features.append(Conv2dNormActivation(
            3,
            first_layer_output,
            kernel_size=3,
            stride=2,
            activation_layer=nn.Hardswish,
        ))

        for k_size, exp_dim, out, SE, Act, s in settings:
            features.append(InvertedResidual(first_layer_output, k_size, exp_dim, out, SE, Act, s, norm_layer))
            first_layer_output = out

        lastconv_output_chann = 6 * first_layer_output

        # Last layer normal convolutional
        features.append(
            Conv2dNormActivation(
                first_layer_output,
                lastconv_output_chann,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        # Classification part
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_chann, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        # Layer initialization
        








