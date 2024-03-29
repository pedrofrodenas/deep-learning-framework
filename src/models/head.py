import torch.nn as nn
from .modules import Activation


# Should be splited from the decoder because we must
# initialize conv2d weights differently because
# they use different activation functions.
class SegmentationHead(nn.Module):
    def __init__(self,
                 classes: int = 1000,
                 activation=None):

        super(SegmentationHead, self).__init__()
        
        self.conv2d = nn.Conv2d(16, classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.activation = Activation(activation)

        # Initialization
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.activation(x)
        return x