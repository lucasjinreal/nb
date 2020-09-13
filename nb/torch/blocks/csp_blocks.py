

from torch import nn
import torch

from .conv_blocks import ConvBase
from ..base import build_norm_layer, build_activation_layer

"""
construct CSP module
most codes comes from yolov5

"""


class SimBottleneck(nn.Module):

    def __init__(self, cin, cout, shortcut=True, g=1, expansion=0.5,
                 norm_cfg=dict(type="BN"), act_cfg=dict(type="Hardswish")):
        super(SimBottleneck, self).__init__()

        c_ = int(cout * expansion)
        self.conv1 = ConvBase(
            cin, c_, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvBase(
            c_, cout, 3, 1, groups=g, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.add = shortcut and cin == cout

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class SimBottleneckCSP(nn.Module):

    """
    Cross Stage Patial network comes from https://github.com/WongKinYiu/CrossStagePartialNetworks

    """

    def __init__(self, cin, cout, n=1, shortcut=True,
                 g=1, expansion=0.5,  norm_cfg=dict(type="BN"), act_cfg=dict(type="Hardswish")):
        super(SimBottleneckCSP, self).__init__()
        c_ = int(cout * expansion)

        self.conv1 = ConvBase(
            cin, c_, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = nn.Conv2d(cin, c_, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)

        self.conv4 = ConvBase(2*c_, cout, 1, 1)

        _, self.bn = build_norm_layer(dict(type='BN'), 2*c_)
        # self.act = build_activation_layer(dict(type='LeakyReLU', inplace=True))
        self.act = build_activation_layer(dict(type='LeakyReLU'))
        self.m = nn.Sequential(
            *[SimBottleneck(c_, c_, shortcut, g, expansion=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.conv3(self.m(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.act(self.bn(torch.cat([y1, y2], dim=1))))
