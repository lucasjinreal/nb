"""

Test mobilenetv3 based on nb
"""

from nb.torch.backbones.mobilenetv3 import mobilenetv3_large

import torch

from nb.torch.backbones.mobilenetv3_new import MobileNetV3_Large



if __name__ == "__main__":
    a = torch.randn([1, 3, 512, 512])
    
    m = mobilenetv3_large(fpn_levels=[5, 9, 14])
    
    o = m(a)
    print(o)
    for i in o:
        print(i.shape)

    m = mobilenetv3_large(num_classes=8)
    
    o = m(a)
    print(o)
    for i in o:
        print(i.shape)
    
    m = MobileNetV3_Large(fpn_levels=[5, 9, 14])
    o = m(a)
    print(o)
    for i in o:
        print(i.shape)