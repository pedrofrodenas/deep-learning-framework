import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from torchvision.ops.misc import Conv2dNormActivation

from typing import List

class InvertedResidual(nn.Module):
    def __init__(self,
                 inp,
                 out, 
                 stride, 
                 expand_ratio):
        
        super(InvertedResidual, self).__init__()

        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))

        # If input channels are equal to input channels we can
        # use residual connection
        self.use_res_connect = self.stride == 1 and inp == out

        # Channel Expansion Layer
        self.bottleneck = Conv2dNormActivation(
            inp,
            hidden_dim,
            kernel_size=1,
            stride=1,
            activation_layer=nn.ReLU6
        )

        # Depth-Wise Convolution 3x3 kernel
        self.depth_wise_conv = Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    activation_layer=nn.ReLU6,
                )
        # Linear projection, (no activation function)
        self.linear_projection = nn.Conv2d(hidden_dim, out, 1, 1, 0, bias=False)

        self.out_channels = out

        layers: List[nn.Module] = []

        if expand_ratio != 1:
            layers.append(self.depth_wise_conv)
        
        layers.extend([self.depth_wise_conv, self.linear_projection])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



class MyMobilenetv2(nn.Module):
    def __init__(self):
        super(MyMobilenetv2, self).__init__()

        input_channel = 32
