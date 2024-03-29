import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from torchvision.ops.misc import Conv2dNormActivation

from typing import List

from .utils import make_divisible

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

        layers: List[nn.Module] = []

        # Channel Expansion Layer
        if expand_ratio != 1:
            layers.append(Conv2dNormActivation(
                inp,
                hidden_dim,
                kernel_size=1,
                stride=1,
                activation_layer=nn.ReLU6
            ))

        # Depth-Wise Convolution 3x3 kernel
        layers.append(Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    activation_layer=nn.ReLU6,
                ))
        # Linear projection, (no activation function)
        layers.append(nn.Conv2d(hidden_dim, out, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out))

        self.out_channels = out

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



class MyMobilenetv2(nn.Module):
    def __init__(self,
                 num_classes: int = 1000,
                 width_mult: float = 1.0,
                 dropout: float = 0.2,
                 pretrained = True,
                 **kwargs):
        super(MyMobilenetv2, self).__init__()

        self.weights_url = "https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth"

        first_layer_output = 32
        last_layer_output = 1280

        # Table2 Mobilenetv2 paper
        settings = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        features: List[nn.Module] = []

        # Width multiplier can reduce or increase the number of
        # input channels but it is necessary to be multiple
        # of 8
        first_layer_output = make_divisible(first_layer_output * width_mult, 8)
        # Also applied to the last layer
        self.last_layer_output = make_divisible(last_layer_output * width_mult, 8)

        # First layer, downsampling / 2
        features.append(Conv2dNormActivation(
            3,
            first_layer_output,
            kernel_size=3,
            stride=2,
            activation_layer=nn.ReLU6,
        ))

        for t, c, n, s in settings:

            output_mod = make_divisible(c * width_mult, 8)
            # Each pattern is repeated n times, with n=0 stride=s and
            # downsampling to spatial dimension is applied, otherwise
            # stride = 1
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(first_layer_output, 
                                                 output_mod,
                                                 stride=stride,
                                                 expand_ratio=t))
                first_layer_output = output_mod

        # Append last layer t = 1, c = 1280, n = 1, s = 1
        features.append(Conv2dNormActivation(first_layer_output,
                                             self.last_layer_output,
                                             kernel_size=1,
                                             stride=1,
                                             activation_layer=nn.ReLU6))
        
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_layer_output, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        #x = self.features(x)

        for f in self.features:
            x = f(x)
            print(x.shape)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    

class Mobilenetv2Backbone(MyMobilenetv2):
    def __init__(self,
                 **kwargs):
        # Se define arquitectura de CustomResNet18 es decir
        # se crea self.state_dict() con la conexion de las capas
        super(Mobilenetv2Backbone, self).__init__()

        # We delete classification layers
        del self.classifier

    def get_forward_outputs(self):
        return [
            self.features[0:2],
            self.features[2:4],
            self.features[4:7],
            self.features[7:14],
            self.features[14:],
        ]
    
    def forward(self, x):

        features = []
        stages = self.get_forward_outputs(x)
        for i in range(len(stages)):
            x = stages[i](x)
            features.append(x)

        return features
    



            


