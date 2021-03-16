

from torch import nn
import torch

from .conv_blocks import ConvBase
from ..base import build_norm_layer, build_activation_layer


class SPPCSP(nn.Module):
    # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSP, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = ConvBase(
            c1, c_, 1, 1, norm_cfg=dict(type="BN"), act_cfg=dict(type="Mish"))
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = ConvBase(c_, c_, 3, 1, norm_cfg=dict(
            type="BN"), act_cfg=dict(type="Mish"))
        self.cv4 = ConvBase(c_, c_, 1, 1, norm_cfg=dict(
            type="BN"), act_cfg=dict(type="Mish"))
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = ConvBase(
            4 * c_, c_, 1, 1, norm_cfg=dict(type="BN"), act_cfg=dict(type="Mish"))
        self.cv6 = ConvBase(c_, c_, 3, 1, norm_cfg=dict(
            type="BN"), act_cfg=dict(type="Mish"))
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = Mish()
        self.cv7 = ConvBase(
            2 * c_, c2, 1, 1, norm_cfg=dict(type="BN"), act_cfg=dict(type="Mish"))

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class BottleneckCSP2(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    # ch_in, ch_out, number, shortcut, groups, expansion
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super(BottleneckCSP2, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = ConvBase(c1, c_, 1, 1, norm_cfg=dict(
            type="BN"), act_cfg=dict(type="Mish"))
        self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv3 = ConvBase(
            2 * c_, c2, 1, 1, norm_cfg=dict(type="BN"), act_cfg=dict(type="Mish"))
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = Mish()
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class VoVCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    # ch_in, ch_out, number, shortcut, groups, expansion
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(VoVCSP, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = ConvBase(
            c1//2, c_//2, 3, 1, norm_cfg=dict(type="BN"), act_cfg=dict(type="Mish"))
        self.cv2 = ConvBase(
            c_//2, c_//2, 3, 1, norm_cfg=dict(type="BN"), act_cfg=dict(type="Mish"))
        self.cv3 = ConvBase(c_, c2, 1, 1, norm_cfg=dict(
            type="BN"), act_cfg=dict(type="Mish"))

    def forward(self, x):
        _, x1 = x.chunk(2, dim=1)
        x1 = self.cv1(x1)
        x2 = self.cv2(x1)
        return self.cv3(torch.cat((x1, x2), dim=1))
