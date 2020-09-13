
import torch
from .conv_blocks import ConvBase


class BasicRFBBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(BasicRFB, self).__init__()
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
