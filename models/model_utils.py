from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn


PROJ_FOLDER = Path(__file__).parents[0]
DATA_DIR = PROJ_FOLDER / "data"


def init_network_weights(net, method='normal_'):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            if method == 'xavier_uniform_':
                nn.init.xavier_uniform_(m.weight)
            elif method == 'kaiming_uniform_':
                nn.init.kaiming_uniform_(m.weight)
            else:
                nn.init.normal_(m.weight, mean=0, std=0.1)

            if m.bias is not None:
                nn.init.constant_(m.bias, val=0)


def sample_standard_gaussian(mu, sigma):
    epsilon = torch.distributions.normal.Normal(torch.Tensor(
        [0.0]).to(mu), torch.Tensor([1.0]).to(mu))
    r = epsilon.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()


def reconstruct_pretrained_flextsf(model, load_para_path, strict=True):
    state_dict = modify_state_dict_keys(
        torch.load(load_para_path)['state_dict'])
    model.load_state_dict(state_dict, strict=strict)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    ft_modules = ["leaderlyr", "revin_layer",
                  "input_linear", "output_linear"]
    for name, param in model.named_parameters():
        for ftm in ft_modules:
            if ftm in name:
                param.requires_grad = True

    return model


def modify_state_dict_keys(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name_parts = k.split('.')
        k = '.'.join(name_parts[1:])
        new_state_dict[k] = v
    return new_state_dict
