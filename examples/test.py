from nb.torch.blocks.asff_blocks import ASFFV5 as ASFF
import torch



if __name__ == "__main__":
    a = torch.randn([1, 128, 64, 64])
    b = torch.randn([1, 256, 32, 32])
    c = torch.randn([1, 512, 16, 16])

    asff0 = ASFF(level=0, multiplier=0.5)
    asff1 = ASFF(level=1, multiplier=0.5)
    asff2 = ASFF(level=2, multiplier=0.5)
    # large -> small
    o = asff0(c, b, a)
    print('-------: ', o.shape)

    o = asff1(c, b, a)
    print('-------: ', o.shape)

    o = asff2(c, b, a)
    print('-------: ', o.shape)

    