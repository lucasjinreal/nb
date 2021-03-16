
from torch import nn as nn

from .conv_ws import ConvWS2d

"""

Since we have various kinds of convs
so we make this function to support different conv type


"""


conv_cfg = {
    'Conv': nn.Conv2d,
    'ConvWS': ConvWS2d,  # https://arxiv.org/pdf/1903.10520.pdf, using when you have a tiny GPU
    # 'DCN': DeformConvPack,  # not supported for now
    # 'DCNv2': ModulatedDeformConvPack,  # not supported for now
    # TODO: octave conv
}


def build_conv_layer(cfg, *args, **kwargs):
    """ Build convolution layer

    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify conv layer type.
            layer args: args needed to instantiate a conv layer.

    Returns:
        layer (nn.Module): created conv layer
    """
    if cfg is None:
        cfg_ = dict(type='Conv')
    else:
        assert isinstance(cfg, dict) and 'type' in cfg
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        conv_layer = conv_cfg[layer_type]

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer
