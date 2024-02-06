import torch
import torch.nn as nn
from torch import Tensor

def resnet_head():
    conv_op = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, padding=3, stride = 2 ,bias=False),
        torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1),
        nn.ReLU(inplace=True)
    )
    return conv_op

def resnet_bottleneck():
    conv_op = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
    )
    return conv_op

class BasicBlock(nn.Module):
    def __init__(self, 
                 input_channel: int,
                 output_channel: int
                ):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1, stride = 1 ,bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel, eps=1e-05, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1, stride = 1 ,bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel, eps=1e-05, momentum=0.1)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class MyResNet32(nn.Module):
    def __init__(self, **kwargs):
        super(MyResNet32, self).__init__()

        self.intro_block = resnet_head()

        # Resolution divided by 2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = BasicBlock(64, 64)
    
    def forward(self, x):

        intro1 = self.intro_block(x)
        max1 = self.maxpool(intro1)
        layer1 = self.layer1(max1)

        return layer1
        

