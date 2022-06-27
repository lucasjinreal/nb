"""
In a [op, c, s, n]
    "MobileOne-S0-Deploy": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [("mobileone", 48, 2, 1, {"over_param_branches": 4}, DEPLOY_CFG)],
            [("mobileone", 48, 2, 2, {"over_param_branches": 4}, DEPLOY_CFG)],
            [("mobileone", 128, 2, 8, {"over_param_branches": 4}, DEPLOY_CFG)],
            [("mobileone", 256, 2, 5, {"over_param_branches": 4}, DEPLOY_CFG)],
            [("mobileone", 256, 1, 5, {"over_param_branches": 4}, DEPLOY_CFG)],
            [("mobileone", 1024, 2, 1, {"over_param_branches": 4}, DEPLOY_CFG)],
            [
                ("adaptive_avg_pool", 1024, 1, 1, {"output_size": 1}),
                ("conv_k1", 1024, 1, 1, {"bias": False}),
            ],
        ],
    },
    "MobileOne-S1-Deploy": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [("mobileone", 96, 2, 1, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 96, 2, 2, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 192, 2, 8, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 512, 2, 5, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 512, 1, 5, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 1280, 2, 1, {"over_param_branches": 1}, DEPLOY_CFG)],
            [
                ("adaptive_avg_pool", 1280, 1, 1, {"output_size": 1}),
                ("conv_k1", 1280, 1, 1, {"bias": False}),
            ],
        ],
    },
    "MobileOne-S2-Deploy": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [("mobileone", 96, 2, 1, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 96, 2, 2, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 256, 2, 8, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 640, 2, 5, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 640, 1, 5, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 2048, 2, 1, {"over_param_branches": 1}, DEPLOY_CFG)],
            [
                ("adaptive_avg_pool", 2048, 1, 1, {"output_size": 1}),
                ("conv_k1", 2048, 1, 1, {"bias": False}),
            ],
        ],
    },
    "MobileOne-S3-Deploy": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [("mobileone", 128, 2, 1, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 128, 2, 2, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 320, 2, 8, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 768, 2, 5, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 768, 1, 5, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 2048, 2, 1, {"over_param_branches": 1}, DEPLOY_CFG)],
            [
                ("adaptive_avg_pool", 2048, 1, 1, {"output_size": 1}),
                ("conv_k1", 2048, 1, 1, {"bias": False}),
            ],
        ],
    },
    # TODO(xfw): Add SE-ReLU in MobileOne-S4
    "MobileOne-S4-Deploy": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [("mobileone", 192, 2, 1, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 192, 2, 2, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 448, 2, 8, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 896, 2, 5, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 896, 1, 5, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 2048, 2, 1, {"over_param_branches": 1}, DEPLOY_CFG)],
            [
                ("adaptive_avg_pool", 2048, 1, 1, {"output_size": 1}),
                ("conv_k1", 2048, 1, 1, {"bias": False}),
            ],
        ],
    },
"""
import torch
from torch.nn import Module
from torch import nn

from nb.torch.backbones.layers.mobileone_block import MobileOneBlock


class MobileOne(Module):
    def __init__(
        self,
        num_classes=1000,
        deploy_mode=False,
        for_classification=True,
    ):
        super(MobileOne, self).__init__()
        self.num_classes = num_classes
        self.for_classification = for_classification

        cfg_s1 = [
            [("mobileone", 96, 2, 1, {"over_param_branches": 1})],
            [("mobileone", 96, 2, 2, {"over_param_branches": 1})],
            [("mobileone", 192, 2, 8, {"over_param_branches": 1})],
            [("mobileone", 512, 2, 5, {"over_param_branches": 1})],
            [("mobileone", 512, 1, 5, {"over_param_branches": 1})],
            [("mobileone", 1280, 2, 1, {"over_param_branches": 1})],
            [
                ("adaptive_avg_pool", 1280, 1, 1, {"output_size": 1}),
                ("conv_k1", 1280, 1, 1, {"bias": False}),
            ],
        ]

        if not for_classification:
            # discard last fc layer
            cfg_s1 = cfg_s1[:-1]

        in_channels = 3
        _blocks = nn.ModuleList([])
        num_block = 0
        for l_cfg in cfg_s1:
            _, c, s, n, _ = l_cfg[0]
            out_channels = c
            for i in range(n):
                _blocks.append(
                    MobileOneBlock(
                        in_channels,
                        out_channels,
                        stride=s,
                        deploy=deploy_mode
                    )
                )
                in_channels = out_channels
                num_block += 1
        self._blocks = _blocks

        if for_classification:
            self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
            self.conv_k1 = nn.Conv2d(1280, 1280, 1)


    def forward(self, x):
        for i, block in enumerate(self._blocks):
            x = block(x)
        if self.for_classification:
            x = self.avg_pool(x)
            x = self.conv_k1(x)
        
        return x

