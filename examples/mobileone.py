from nb.torch.backbones.mobileone import MobileOne
import torch


a = MobileOne(deploy_mode=True)

x = torch.randn(2, 3, 224, 224)
print(a)

o = a(x)

print(o.shape)