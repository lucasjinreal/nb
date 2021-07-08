import torch.nn as nn
import torch.nn.functional as F
import torch
import logging


try:
    from mish_cuda import MishCuda as Mish
except Exception:
    logger = logging.getLogger(__name__)
    logger.warning(
        "Install mish-cuda to speed up training and inference. More "
        "importantly, replace the naive Mish with MishCuda will give a "
        "~1.5G memory saving during training."
    )

    def mish(x):
        return x.mul(F.softplus(x).tanh())

    class Mish(nn.Module):
        def __init__(self):
            super(Mish, self).__init__()

        def forward(self, x):
            return mish(x)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


activation_cfg = {
    # layer_abbreviation: module
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'PReLU': nn.PReLU,
    'RReLU': nn.RReLU,
    'ReLU6': nn.ReLU6,
    'SELU': nn.SELU,
    'CELU': nn.CELU,
    # new added
    'Hardswish': nn.Hardswish,  # check pytorch version, >= 1.6
    'SiLU': nn.SiLU,  # check pytorch version, >= 1.7
    'Mish': Mish
}


def build_activation_layer(cfg):
    """ Build activation layer

    Args:
        cfg (dict): cfg should contain:
            type (str): Identify activation layer type.
            layer args: args needed to instantiate a activation layer.

    Returns:
        layer (nn.Module): Created activation layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in activation_cfg:
        raise KeyError('Unrecognized activation type {}'.format(layer_type))
    else:
        activation = activation_cfg[layer_type]
        if activation is None:
            raise NotImplementedError
    layer = activation(**cfg_)
    return layer
