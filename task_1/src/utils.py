import copy
import json
import random
import re

import numpy as np
import torch
import yaml


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_by_path(config, dotted_path, value):
    keys = dotted_path.split(".")
    node = config
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value


def apply_overrides(base_config, overrides):
    config = copy.deepcopy(base_config)
    for dotted_path, value in overrides.items():
        set_by_path(config, dotted_path, value)
    return config


def load_color_mapping(json_path, classes):
    """Parses color-mappings.json, tolerating the '// Ignore' style trailing
    comments used in some dataset files (not valid strict JSON)."""
    with open(json_path, "r") as f:
        raw = f.read()
    raw = re.sub(r"//.*", "", raw)
    data = json.loads(raw)
    mapping = {}
    for cls in classes:
        hex_color = data[cls].lstrip("#")
        mapping[cls] = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return mapping
