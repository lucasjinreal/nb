
from nb.torch.blocks.csp_blocks import SimBottleneckCSP

import torch



if __name__ == "__main__":
    a = torch.randn([1, 1024, 256, 256])
    m = SimBottleneckCSP(1024, 512)

    o = m(a)

    print(o)
    print(o.shape)

