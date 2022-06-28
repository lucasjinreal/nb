import torch

def load_ckp_unwrap_module(ckp_p):
    sd = torch.load(ckp_p, map_location='cpu')

    state_dict = None
    if 'state_dict' in sd.keys():
        state_dict = sd['state_dict']
    elif 'model' in sd.keys():
        state_dict = sd['model']
    else:
        state_dict = sd

    new_sd = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            new_sd[k.replace('module.', '')] = v
        else:
            new_sd[k] = v
    return new_sd