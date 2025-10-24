import torch
import copy
from experiments.common.conv2d_img2col import Conv2dImg2Col
import torch.nn as nn


class Conv2dImg2ColWithBN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=False,
        replaced_conv=False,
    ):
        super().__init__()
        if replaced_conv:

            self.conv = Conv2dImg2Col(
                in_channels, out_channels, kernel_size, stride, padding, bias
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=bias
            )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


# Dummy model with batch normalization
class DummyModel(nn.Module):
    def __init__(self, replaced_conv=False):
        super().__init__()
        self.replaced_conv = replaced_conv

        # Use fused Conv-BN blocks
        self.conv1 = Conv2dImg2ColWithBN(3, 16, kernel_size=3, stride=1, padding=1, replaced_conv=replaced_conv)
        self.relu1 = nn.ReLU()
        self.conv2 = Conv2dImg2ColWithBN(16, 32, kernel_size=3, stride=1, padding=1, replaced_conv=replaced_conv)
        self.relu2 = nn.ReLU()
        self.conv3 = Conv2dImg2ColWithBN(32, 64, kernel_size=3, stride=1, padding=1, replaced_conv=replaced_conv)
        self.relu3 = nn.ReLU()

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)  # 10 classes for classification

    def forward(self, x):

        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
