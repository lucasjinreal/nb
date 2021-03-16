

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



class BasicRFBBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(BasicRFBBlock, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8  # inter_planes=128 ,in_planes=1024
        self.branch0 = nn.Sequential(
            ConvBase(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
            ConvBase(2*inter_planes, 2*inter_planes, kernel_size=3,
                     stride=1, padding=visual, dilation=visual, act_cfg=None)
        )
        self.branch1 = nn.Sequential(
            ConvBase(in_planes, inter_planes, kernel_size=1, stride=1),
            ConvBase(inter_planes, 2*inter_planes, kernel_size=(3, 3),
                     stride=stride, padding=(1, 1)),
            ConvBase(2*inter_planes, 2*inter_planes, kernel_size=3,
                     stride=1, padding=visual+1, dilation=visual+1, act_cfg=None)
        )
        self.branch2 = nn.Sequential(
            ConvBase(in_planes, inter_planes, kernel_size=1, stride=1),
            ConvBase(inter_planes, (inter_planes//2)*3,
                     kernel_size=3, stride=1, padding=1),
            ConvBase((inter_planes//2)*3, 2*inter_planes,
                     kernel_size=3, stride=stride, padding=1),
            ConvBase(2*inter_planes, 2*inter_planes, kernel_size=3,
                     stride=1, padding=2*visual+1, dilation=2*visual+1, act_cfg=None)
        )

        self.ConvLinear = ConvBase(
            6*inter_planes, out_planes, kernel_size=1, stride=1, act_cfg=None)
        self.shortcut = ConvBase(
            in_planes, out_planes, kernel_size=1, stride=stride, act_cfg=None)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out