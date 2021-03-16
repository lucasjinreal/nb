

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .conv_blocks import ConvBase

"""

ASFF blocks which better than RFBnet

code token from: https://github.com/ruinmessi/ASFF/blob/master/models/network_blocks.py
with some our own modifications

"""


class ASFFmobile(nn.Module):
    def __init__(self, level, rfb=False, vis=False, act_cfg=dict(type='ReLU6')):
        super(ASFFmobile, self).__init__()
        self.level = level
        self.dim = [512, 256, 128]
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = ConvBase(
                256, self.inter_dim, 3, 2, act_cfg=act_cfg)
            self.stride_level_2 = ConvBase(
                128, self.inter_dim, 3, 2, act_cfg=act_cfg)
            self.expand = ConvBase(self.inter_dim, 1024, 3, 1, act_cfg=act_cfg)
        elif level == 1:
            self.compress_level_0 = ConvBase(
                512, self.inter_dim, 1, 1, act_cfg=act_cfg)
            self.stride_level_2 = ConvBase(
                128, self.inter_dim, 3, 2, act_cfg=act_cfg)
            self.expand = ConvBase(self.inter_dim, 512, 3, 1, act_cfg=act_cfg)
        elif level == 2:
            self.compress_level_0 = ConvBase(
                512, self.inter_dim, 1, 1, act_cfg=act_cfg)
            self.compress_level_1 = ConvBase(
                256, self.inter_dim, 1, 1, act_cfg=act_cfg)
            self.expand = ConvBase(self.inter_dim, 256, 3, 1, act_cfg=act_cfg)

        # when adding rfb, we use half number of channels to save memory
        compress_c = 8 if rfb else 16

        self.weight_level_0 = ConvBase(
            self.inter_dim, compress_c, 1, 1, act_cfg=act_cfg)
        self.weight_level_1 = ConvBase(
            self.inter_dim, compress_c, 1, 1, act_cfg=act_cfg)
        self.weight_level_2 = ConvBase(
            self.inter_dim, compress_c, 1, 1, act_cfg=act_cfg)

        self.weight_levels = nn.Conv2d(
            compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter = F.max_pool2d(
                x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=4, mode='nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(
                level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] +\
            level_1_resized * levels_weight[:, 1:2, :, :] +\
            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class ASFF(nn.Module):
    def __init__(self, level, multiplier=1, rfb=False, vis=False, act_cfg=dict(type='LeakyReLU')):
        """
        normally, multiplier should be 1, 0.5
        which means, the channel of ASFF can be 
        512, 256, 256 -> multiplier=1
        256, 128, 128 -> multiplier=0.5
        For even smaller, you gonna need change code manually.
        If you got any question about this, consult me via wechat: jintianiloveu
        """
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [int(512*multiplier), int(256*multiplier),
                    int(256*multiplier)]
        # print(self.dim)
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = ConvBase(
                int(256*multiplier), self.inter_dim, 3, 2, act_cfg=act_cfg)
            self.stride_level_2 = ConvBase(
                int(256*multiplier), self.inter_dim, 3, 2, act_cfg=act_cfg)
            self.expand = ConvBase(self.inter_dim, int(
                1024*multiplier), 3, 1, act_cfg=act_cfg)
        elif level == 1:
            self.compress_level_0 = ConvBase(
                int(512*multiplier), self.inter_dim, 1, 1, act_cfg=act_cfg)
            self.stride_level_2 = ConvBase(
                int(256*multiplier), self.inter_dim, 3, 2, act_cfg=act_cfg)
            self.expand = ConvBase(self.inter_dim, int(512*multiplier), 3, 1)
        elif level == 2:
            self.compress_level_0 = ConvBase(
                int(512*multiplier), self.inter_dim, 1, 1, act_cfg=act_cfg)
            self.expand = ConvBase(self.inter_dim, int(
                256*multiplier), 3, 1, act_cfg=act_cfg)

        # when adding rfb, we use half number of channels to save memory
        compress_c = 8 if rfb else 16

        self.weight_level_0 = ConvBase(
            self.inter_dim, compress_c, 1, 1, act_cfg=act_cfg)
        self.weight_level_1 = ConvBase(
            self.inter_dim, compress_c, 1, 1, act_cfg=act_cfg)
        self.weight_level_2 = ConvBase(
            self.inter_dim, compress_c, 1, 1, act_cfg=act_cfg)

        self.weight_levels = nn.Conv2d(
            compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2):
        """
        from small -> large
        """
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter = F.max_pool2d(
                x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=4, mode='nearest')
            level_1_resized = F.interpolate(
                x_level_1, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] +\
            level_1_resized * levels_weight[:, 1:2, :, :] +\
            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class ASFFV5(nn.Module):
    def __init__(self, level, multiplier=1, rfb=False, vis=False, act_cfg=dict(type='LeakyReLU')):
        """
        this is ASFF version for YoloV5 only.
        Since YoloV5 outputs 3 layer of feature maps with different channels
        which is different than YoloV3

        normally, multiplier should be 1, 0.5
        which means, the channel of ASFF can be 
        512, 256, 128 -> multiplier=1
        256, 128, 64 -> multiplier=0.5
        For even smaller, you gonna need change code manually.
        If you got any question about this, consult me via wechat: jintianiloveu
        """
        super(ASFFV5, self).__init__()
        self.level = level
        self.dim = [int(1024*multiplier), int(512*multiplier),
                    int(256*multiplier)]
        # print(self.dim)
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = ConvBase(
                int(512*multiplier), self.inter_dim, 3, 2, act_cfg=act_cfg)
            self.stride_level_2 = ConvBase(
                int(256*multiplier), self.inter_dim, 3, 2, act_cfg=act_cfg)
            self.expand = ConvBase(self.inter_dim, int(
                1024*multiplier), 3, 1, act_cfg=act_cfg)
        elif level == 1:
            self.compress_level_0 = ConvBase(
                int(1024*multiplier), self.inter_dim, 1, 1, act_cfg=act_cfg)
            self.stride_level_2 = ConvBase(
                int(256*multiplier), self.inter_dim, 3, 2, act_cfg=act_cfg)
            self.expand = ConvBase(self.inter_dim, int(512*multiplier), 3, 1)
        elif level == 2:
            self.compress_level_0 = ConvBase(
                int(1024*multiplier), self.inter_dim, 1, 1, act_cfg=act_cfg)
            self.compress_level_1 = ConvBase(
                int(512*multiplier), self.inter_dim, 1, 1, act_cfg=act_cfg)
            self.expand = ConvBase(self.inter_dim, int(
                256*multiplier), 3, 1, act_cfg=act_cfg)

        # when adding rfb, we use half number of channels to save memory
        compress_c = 8 if rfb else 16

        self.weight_level_0 = ConvBase(
            self.inter_dim, compress_c, 1, 1, act_cfg=act_cfg)
        self.weight_level_1 = ConvBase(
            self.inter_dim, compress_c, 1, 1, act_cfg=act_cfg)
        self.weight_level_2 = ConvBase(
            self.inter_dim, compress_c, 1, 1, act_cfg=act_cfg)

        self.weight_levels = nn.Conv2d(
            compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2):
        """
        # 128, 256, 512

        512, 256, 128
        from small -> large
        """
        # print('x_level_0: ', x_level_0.shape)
        # print('x_level_1: ', x_level_1.shape)
        # print('x_level_2: ', x_level_2.shape)
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter = F.max_pool2d(
                x_level_2, 3, stride=2, padding=1)
            # print('128 ', level_2_downsampled_inter.shape)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(
                x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        # print('level: {}, l1_resized: {}, l2_resized: {}'.format(self.level,
            #  level_1_resized.shape, level_2_resized.shape))
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        # print('level_0_weight_v: ', level_0_weight_v.shape)
        # print('level_1_weight_v: ', level_1_weight_v.shape)
        # print('level_2_weight_v: ', level_2_weight_v.shape)

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] +\
            level_1_resized * levels_weight[:, 1:2, :, :] +\
            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out
