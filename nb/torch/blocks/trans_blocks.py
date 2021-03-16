from torch import nn
import torch

from .conv_blocks import ConvBase
from ..base import build_norm_layer, build_activation_layer


"""

Transform space information here

"""


class Focus(nn.Module):
    # Focus wh information into c-space
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        self.conv = ConvBase(c1*4, c2, k, s, p, g,
                             act_cfg=dict(type="Hardswish"))

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            ConvBase(in_channels, out_channels,
                     1, act_cfg=dict(type='LeakyReLU')), nn.Upsample(scale_factor=scale)
        )

    def forward(self, x):
        return self.upsample(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(Downsample, self).__init__()
        self.downsample = ConvBase(
            in_channels, out_channels, 3, 2, act_cfg=dict(type='LeakyReLU'))

    def forward(self, x):
        return self.downsample(x)
