"""
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


class MobileOne(Module):
    def __init__(self,
                 width_mult=1.0,
                 depth_mult=1.0,
                 dropout_rate=0.2,
                 num_classes=1000,
                 features_indices=[1, 4, 10, 15],
                 bn_mom=0.99,
                 bn_eps=1e-3
                 ):
        super(MobileOne, self).__init__()
        self.num_classes = num_classes
        self.extract_features = num_classes <= 0
        # stride=2:  ----> block 1 ,3, 5 ,11
        self.return_features_indices = features_indices
        out_feature_channels = []
        out_feature_strides = [4, 8, 16, 32]
       

    def forward(self, x):
        pass

   
