from nb.torch.backbones.mobileone import MobileOneNet, make_mobileone_s0
from nb.torch.utils.checkpoint import load_ckp_unwrap_module
import sys
import torch
import os

a = sys.argv[1]

sd = load_ckp_unwrap_module(a)

x = torch.randn(1, 3, 224, 224)

model = make_mobileone_s0(deploy=False)
model.load_state_dict(sd)
print("original model loaded.")

for module in model.modules():
    if hasattr(module, "switch_to_deploy"):
        module.switch_to_deploy()

o1 = model(x)

deploy_model = make_mobileone_s0(deploy=True)
deploy_model.eval()
deploy_model.load_state_dict(model.state_dict())
o = deploy_model(x)

print((o1 - o).sum())

n_f = os.path.join(
    os.path.dirname(a), os.path.basename(a).split(".")[0] + "_reparam.pth"
)
torch.save(model.state_dict(), n_f)

mod = torch.jit.trace(deploy_model, x)

n_f2 = os.path.join(
    os.path.dirname(a), os.path.basename(a).split(".")[0] + "_reparam.pt"
)
mod.save(n_f2)