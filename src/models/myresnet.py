import torch
import torch.nn as nn
from torch import Tensor

def resnet_downsampling(input_channels, output_channels):
    conv_op = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(output_channels, eps=1e-05, momentum=0.1)
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
    

class DownsamplingBlock(nn.Module):
    def __init__(self, 
                 input_channel: int,
                 output_channel: int
                ):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1, stride = 2 ,bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel, eps=1e-05, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1, stride = 1 ,bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel, eps=1e-05, momentum=0.1)
        self.downsample = resnet_downsampling(input_channel, output_channel)

    def forward(self, x: Tensor) -> Tensor:
        # identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(x)
        out = self.relu(out)
        return out


class MyResNet18(nn.Module):
    def __init__(self,
                 classes: int = 1000,
                 zero_init_residual: bool = False,
                 **kwargs):
        super(MyResNet18, self).__init__()

        # ResNet Head
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3, stride = 2 ,bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

        # Resolution divided by 2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(DownsamplingBlock(64, 128), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(DownsamplingBlock(128, 256), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(DownsamplingBlock(256, 512), BasicBlock(512, 512))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 , classes)

        # Weights Initialization
        # fan_out because I assume that if you suspect your backward pass 
        # might be more "chaotic" (greater variance) it is worth changing 
        # the mode to fan_out. This might happen when the loss oscillates 
        # a lot (e.g. very easy examples followed by very hard ones).
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, DownsamplingBlock) and m.downsampling[1].weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        # Weights initialization
        state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', weights_only = False)

        super().load_state_dict(state_dict)
    
    def forward(self, x: Tensor) -> Tensor:

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
        

