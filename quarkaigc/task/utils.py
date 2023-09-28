# -*- coding: utf-8 -*-
# @Time   : 2023/5/10 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import warnings
import importlib

warnings.filterwarnings(action='ignore', category=UserWarning)

def instantiate_data_from_config(config):
    module, cls = config["lightningModule"]["target"], "DataFactoryModule"
    data_conf = config["lightningModule"]["data"]
    importlib.invalidate_caches()
    module_imp = getattr(importlib.import_module(module, package=None), cls)
    return module_imp(data_conf)

def instantiate_task_from_config(config):
    module, cls = config["lightningModule"]["target"], "TaskFactoryModule"
    model_conf = config["lightningModule"]["model"]
    importlib.invalidate_caches()
    module_imp = getattr(importlib.import_module(module, package=None), cls)
    return module_imp(model_conf)

def instantiate_model_card_from_config(config):
    module, cls = config["target"].rsplit(".", 1)
    model_card = config["model_card_id"]
    cache_dir = config.get("cache_dir", "./")

    importlib.invalidate_caches()
    module_imp = getattr(importlib.import_module(module, package=None), cls)
    return module_imp.from_pretrained(model_card, cache_dir=cache_dir)


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def instantiate_pretrained_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"]).from_pretrained(
        config["model_type"])


def instantiate_optimizer_from_config(params, config):
    return get_obj_from_str(config["target"])(
        params,
        **config.get("params", dict()),
    )


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
