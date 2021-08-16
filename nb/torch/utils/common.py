
import torch
import math

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor
