
import torch
from torch import nn

from nb.torch.blocks.bottleneck_blocks import SimBottleneckCSP
from nb.torch.blocks.trans_blocks import Focus
from nb.torch.blocks.head_blocks import SPP
from nb.torch.blocks.conv_blocks import ConvBase

from nb.torch.utils import device


class YoloV5(nn.Module):

    def __init__(self, num_cls=80, ch=3, anchors=None):
        super(YoloV5, self).__init__()
        assert anchors != None, 'anchor must be provided'

        # divid by
        cd = 2
        wd = 3

        self.focus = Focus(ch, 64//cd)
        self.conv1 = ConvBase(64//cd, 128//cd, 3, 2)
        self.csp1 = SimBottleneckCSP(128//cd, 128//cd, n=3//wd)
        self.conv2 = ConvBase(128//cd, 256//cd, 3, 2)
        self.csp2 = SimBottleneckCSP(256//cd, 256//cd, n=9//wd)
        self.conv3 = ConvBase(256//cd, 512//cd, 3, 2)
        self.csp3 = SimBottleneckCSP(512//cd, 512//cd, n=9//wd)
        self.conv4 = ConvBase(512//cd, 1024//cd, 3, 2)
        self.spp = SPP(1024//cd, 1024//cd)
        self.csp4 = SimBottleneckCSP(1024//cd, 1024//cd, n=3//wd, shortcut=False)

        # PANet
        self.conv5 = ConvBase(1024//cd, 512//cd)
        self.up1 = nn.Upsample(scale_factor=2)
        self.csp5 = SimBottleneckCSP(1024//cd, 512//cd, n=3//wd, shortcut=False)

        self.conv6 = ConvBase(512//cd, 256//cd)
        self.up2 = nn.Upsample(scale_factor=2)
        self.csp6 = SimBottleneckCSP(512//cd, 256//cd, n=3//wd, shortcut=False)

        self.conv7 = ConvBase(256//cd, 256//cd, 3, 2)
        self.csp7 = SimBottleneckCSP(512//cd, 512//cd, n=3//wd, shortcut=False)

        self.conv8 = ConvBase(512//cd, 512//cd, 3, 2)
        self.csp8 = SimBottleneckCSP(512//cd, 1024//cd, n=3//wd, shortcut=False)

    def _build_backbone(self, x):
        x = self.focus(x)
        x = self.conv1(x)
        x = self.csp1(x)
        x_p3 = self.conv2(x)  # P3
        x = self.csp2(x_p3)
        x_p4 = self.conv3(x)  # P4
        x = self.csp3(x_p4)
        x_p5 = self.conv4(x)  # P5
        x = self.spp(x_p5)
        x = self.csp4(x)
        return x_p3, x_p4, x_p5, x

    def _build_head(self, p3, p4, p5, feas):
        h_p5 = self.conv5(feas)  # head P5
        x = self.up1(h_p5)
        x_concat = torch.cat([x, p4], dim=1)
        x = self.csp5(x_concat)

        h_p4 = self.conv6(x)  # head P4
        x = self.up2(h_p4)
        x_concat = torch.cat([x, p3], dim=1)
        x_small = self.csp6(x_concat)

        x = self.conv7(x_small)
        x_concat = torch.cat([x, h_p4], dim=1)
        x_medium = self.csp7(x_concat)

        x = self.conv8(x_medium)
        x_concat = torch.cat([x, h_p5], dim=1)
        x_large = self.csp8(x)
        return x_small, x_medium, x_large

    def forward(self, x):
        p3, p4, p5, feas = self._build_backbone(x)
        xs, xm, xl = self._build_head(p3, p4, p5, feas)
        return xs, xm, xl


if __name__ == "__main__":
    a = torch.randn([1, 3, 512, 512]).to(device)

    anchors = [[4, 6,  5, 12,  8, 8],
               [13, 12,  8, 20,  13, 31],
               [32, 20,  18, 42,  28, 59]]
    model = YoloV5(anchors=anchors).to(device)

    o = model(a)
    for a in o:
        print(a.shape)
    print('this is the output of yolov5 to be sent to Detect layer')
