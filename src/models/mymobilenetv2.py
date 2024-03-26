import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from torchvision.ops.misc import Conv2dNormActivation

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

        # Depth-Wise Convolution
        self.depth_wise_conv = Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    activation_layer=nn.ReLU6,
                )
        self.linear_projection = nn.Conv2d(hidden_dim, out, 1, 1, 0, bias=False)





class MyMobilenetv2(nn.Module):
    def __init__(self):
        super(MyMobilenetv2, self).__init__()

        input_channel = 32
