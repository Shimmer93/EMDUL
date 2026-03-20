import yaml
from argparse import Namespace
import sys
from collections import OrderedDict

def load_cfg(cfg):
    if isinstance(cfg, str):
        with open(cfg, errors='ignore') as f:
            hyp = yaml.safe_load(f)
        return Namespace(**hyp)
    if isinstance(cfg, dict):
        return Namespace(**cfg)
    raise TypeError(f'Unsupported config type: {type(cfg).__name__}')


def merge_args_cfg(args, cfg):
    args_dict = vars(args)
    cfg_dict = vars(cfg)
    merged = {**args_dict, **cfg_dict}
    return Namespace(**merged)

def torch2numpy(tensor):
    return tensor.detach().cpu().numpy()

def import_with_str(module, name):
    return getattr(sys.modules[module], name)

def delete_prefix_from_state_dict(state_dict, prefix):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def exists_and_is_true(hparams, key):
    return hasattr(hparams, key) and getattr(hparams, key)
