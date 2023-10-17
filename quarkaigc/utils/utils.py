# -*- coding: utf-8 -*-
# @Time   : 2023/5/10 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import warnings
import importlib
from typing import Dict

warnings.filterwarnings(action='ignore', category=UserWarning)


def instantiate_from_model_card(config):
    module, cls = config["target"].rsplit(".", 1)
    model_card = config["model_card_id"]
    cache_dir = config.get("cache_dir", "./")
    subfolder = config.get("subfolder", None)
    local_files_only = config.get("local_files_only", False)

    importlib.invalidate_caches()
    module_imp = getattr(importlib.import_module(module, package=None), cls)
    if subfolder:
        return module_imp.from_pretrained(
            model_card,
            subfolder=subfolder,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    else:
        return module_imp.from_pretrained(
            model_card,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def instantiate_from_params_config(params, config):
    return get_obj_from_str(config["target"])(
        params,
        **config.get("params", dict()),
    )


def instantiate_from_params(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(config.get("params", dict()))


def instantiate_from_one_params(params, config):
    return get_obj_from_str(config["target"])(
        params,
        config.get("params", dict()),
    )


def instantiate_from_two_params(params01, params02, config):
    return get_obj_from_str(config["target"])(
        params01,
        params02,
        config.get("params", dict()),
    )


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
