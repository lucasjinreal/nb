from torch import nn
import torch

from .conv_blocks import ConvBase
from ..base import build_norm_layer, build_activation_layer
from .trans_blocks import Downsample, Upsample


"""

We introduce head blocks here, such as:

PANet
SPP
FPN

etc.

"""


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = ConvBase(c1, c_, 1, 1)
        self.cv2 = ConvBase(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))





class PANet(nn.Module):

    """
    it takes input as:
    torch.Size([1, 256, 64, 128])
    torch.Size([1, 512, 32, 64])
    torch.Size([1, 2048, 16, 32])

    outputs:
    torch.Size([1, 128, 64, 128])
    torch.Size([1, 256, 32, 64])
    torch.Size([1, 512, 16, 32])

    it actually down-sized channel of input

    """

    def __init__(self, feature_channels):
        super(PANet, self).__init__()

        self.feature_transform3 = ConvBase(
            feature_channels[0], feature_channels[0] // 2, 1)
        self.feature_transform4 = ConvBase(
            feature_channels[1], feature_channels[1] // 2, 1)

        self.resample5_4 = Upsample(
            feature_channels[2] // 2, feature_channels[1] // 2)
        self.resample4_3 = Upsample(
            feature_channels[1] // 2, feature_channels[0] // 2)
        self.resample3_4 = Downsample(
            feature_channels[0] // 2, feature_channels[1] // 2)
        self.resample4_5 = Downsample(
            feature_channels[1] // 2, feature_channels[2] // 2)

        self.downstream_conv5 = nn.Sequential(
            ConvBase(feature_channels[2] * 2, feature_channels[2] // 2, 1),
            ConvBase(feature_channels[2] // 2, feature_channels[2], 3),
            ConvBase(feature_channels[2], feature_channels[2] // 2, 1),)
        self.downstream_conv4 = nn.Sequential(
            ConvBase(feature_channels[1], feature_channels[1] // 2, 1),
            ConvBase(feature_channels[1] // 2, feature_channels[1], 3),
            ConvBase(feature_channels[1], feature_channels[1] // 2, 1),
            ConvBase(feature_channels[1] // 2, feature_channels[1], 3),
            ConvBase(feature_channels[1], feature_channels[1] // 2, 1),)
        self.downstream_conv3 = nn.Sequential(
            ConvBase(feature_channels[0], feature_channels[0] // 2, 1),
            ConvBase(feature_channels[0] // 2, feature_channels[0], 3),
            ConvBase(feature_channels[0], feature_channels[0] // 2, 1),
            ConvBase(feature_channels[0] // 2, feature_channels[0], 3),
            ConvBase(feature_channels[0], feature_channels[0] // 2, 1),)

        self.upstream_conv4 = nn.Sequential(
            ConvBase(feature_channels[1], feature_channels[1] // 2, 1),
            ConvBase(feature_channels[1] // 2, feature_channels[1], 3),
            ConvBase(feature_channels[1], feature_channels[1] // 2, 1),
            ConvBase(feature_channels[1] // 2, feature_channels[1], 3),
            ConvBase(feature_channels[1], feature_channels[1] // 2, 1),)
        self.upstream_conv5 = nn.Sequential(
            ConvBase(feature_channels[2], feature_channels[2] // 2, 1),
            ConvBase(feature_channels[2] // 2, feature_channels[2], 3),
            ConvBase(feature_channels[2], feature_channels[2] // 2, 1),
            ConvBase(feature_channels[2] // 2, feature_channels[2], 3),
            ConvBase(feature_channels[2], feature_channels[2] // 2, 1),)
        self._initialize_weights()

    def forward(self, features):
        features = [
            self.feature_transform3(features[0]),
            self.feature_transform4(features[1]),
            features[2], ]
        # for i in features:
        # print('PAN: ', i.shape)

        downstream_feature5 = self.downstream_conv5(features[2])
        downstream_feature4 = self.downstream_conv4(
            torch.cat(
                [features[1], self.resample5_4(downstream_feature5)], dim=1
            ))
        downstream_feature3 = self.downstream_conv3(
            torch.cat(
                [features[0], self.resample4_3(downstream_feature4)], dim=1
            ))
        upstream_feature4 = self.upstream_conv4(
            torch.cat(
                [self.resample3_4(downstream_feature3), downstream_feature4],
                dim=1,
            ))
        upstream_feature5 = self.upstream_conv5(
            torch.cat(
                [self.resample4_5(upstream_feature4), downstream_feature5],
                dim=1,
            ))
        return [downstream_feature3, upstream_feature4, upstream_feature5]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
