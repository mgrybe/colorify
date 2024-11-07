import torch
import torch.nn as nn
import torch.nn.utils as utils

from fastai.layers import SelfAttention


# Source: https://github.com/znxlwm/pytorch-pix2pix/blob/3059f2af53324e77089bbcfc31279f01a38c40b8/network.py#L104
class PatchDiscriminator(nn.Module):

    def __init__(self, ni):
        super(PatchDiscriminator, self).__init__()

        self.conv1 = self.conv_block(in_channels=ni, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv2 = self.conv_block(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv3 = self.conv_block(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv4 = self.conv_block(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1)
        self.conv5 = self.conv_block(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, act=False)

    def conv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, act=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        conv = utils.spectral_norm(conv)

        layer = [conv]

        if act:
            activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            layer += [activation]

        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


