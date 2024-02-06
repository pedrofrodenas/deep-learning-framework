import torch
import torch.nn as nn

def resnet_head():
    conv_op = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, padding=3, stride = 2 ,bias=False),
        torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1),
        nn.ReLU(inplace=True)
    )
    return conv_op

def resnet_block():
    conv_op = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=3, padding=1, stride = 1 ,bias=False),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1, stride = 1 ,bias=False),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1),
    )
    return conv_op

def resnet_bottleneck():
    conv_op = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
    )
    return conv_op

class MyResNet32(nn.Module):
    def __init__(self, **kwargs):
        super(MyResNet32, self).__init__()

        self.intro_block = resnet_head()

        # Resolution divided by 2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.resnet_block = resnet_block()
    
    def forward(self, x):

        intro1 = self.intro_block(x)
        max1 = self.maxpool(intro1)
        layer1 = self.resnet_block(max1)
        layer2 = self.resnet_block(layer1)

        return layer2
        

