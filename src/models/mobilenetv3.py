from typing import List
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation

from .decoders import DecoderBlock
from .head import SegmentationHead
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
        # It uses ReLU , not ReLU6 as described in the original paper
        # it also uses Conv2D rather than fully-connected layers
        if SE:
            squeeze_channels = make_divisible(hidden // 4, 8)
            layers.append(SqueezeExcitation(hidden, squeeze_channels, scale_activation=nn.Hardsigmoid))

        # Linear projection, (no activation function)
        layers.append(
            Conv2dNormActivation(hidden, out, kernel_size=1, norm_layer=norm_layer, 
                                 activation_layer=None)
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


class Mobilenetv3_small(nn.Module):
    def __init__(self,
                 num_classes: int = 1000,
                 width_mult: float = 1.0,
                 dropout: float = 0.2,
                 pretrained = True,
                 **kwargs):
        super(Mobilenetv3_small, self).__init__()

        first_layer_output = 16
        last_channel = 1024
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        self.weights_url = "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"

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

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Load pretrained weights
        if pretrained:
            # Weights initialization
            state_dict = torch.hub.load_state_dict_from_url(self.weights_url)
            super().load_state_dict(state_dict)
        
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    

class Mobilenetv3_small_Backbone(Mobilenetv3_small):
    def __init__(self,
                 **kwargs):
        super(Mobilenetv3_small_Backbone, self).__init__()

        # We delete classification layers
        del self.classifier
        del self.avgpool

    def get_forward_outputs(self):
        return [
            self.features[0],
            self.features[1],
            self.features[2:4],
            self.features[4:9],
            self.features[9:],
        ]
    
    def forward(self, x):

        features = []
        stages = self.get_forward_outputs()
        for i in range(len(stages)):
            x = stages[i](x)
            features.append(x)
        return features
    
class Mobilenetv3_small_Decoder(nn.Module):
    def __init__(self):
        super(Mobilenetv3_small_Decoder, self).__init__()

        # out_channels = [256, 128, 64, 32, 16] estos son fijos
        # input channels are number of channels of features from encoder + cocatenation of
        # previous layer output
        self.block0 = DecoderBlock(624, 256)
        self.block1 = DecoderBlock(280, 128)
        self.block2 = DecoderBlock(144, 64)
        self.block3 = DecoderBlock(80, 32)
        # Last block has no concatenation from encoder, input channel equals last block
        self.block4 = DecoderBlock(32, 16)

        # Layer initialization
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, *features):
    
        x = self.block0(features[4], features[3])
        x = self.block1(x, features[2])
        x = self.block2(x, features[1])
        x = self.block3(x, features[0])
        x = self.block4(x)
        return x

class Mobilenetv3_small_Segmentation(nn.Module):
    def __init__(self,
                 classes: int = 1000,
                 activation=None,
                 **kwargs):
        super(Mobilenetv3_small_Segmentation, self).__init__()

        self.mobilenetv3_encoder = Mobilenetv3_small_Backbone()
        self.mobilenetv3_decoder = Mobilenetv3_small_Decoder()
        self.segmentation_head = SegmentationHead(classes, input_chn = 16, activation = activation)

    def forward(self, x):
        features = self.mobilenetv3_encoder(x)
        decoder_output = self.mobilenetv3_decoder(*features)
        head_output = self.segmentation_head(decoder_output)
        return head_output







