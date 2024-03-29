import torch.nn as nn
import torch
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self,
                 input_channel: int,
                 output_channel: int
                 ):
        super(DecoderBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel, eps=1e-05, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel, eps=1e-05, momentum=0.1)

    def forward(self, x, feature=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if feature is not None:
            x = torch.cat((x, feature), dim = 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x