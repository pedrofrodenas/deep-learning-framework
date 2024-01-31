import torch
import torch.nn as nn

def resnet_head():
    conv_op = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, padding=3, stride = 2 ,bias=False),
        torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1),
        nn.ReLU(inplace=True)
    )
    return conv_op

class MyResNet32(nn.Module):
    def __init__(self):
        super(MyResNet32, self).__init__()

        self.intro_block = resnet_head(3, 64)
        

