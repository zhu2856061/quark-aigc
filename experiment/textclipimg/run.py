# -*- coding: utf-8 -*-
# @Time   : 2023/8/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function
import sys

sys.path.append("../..")

from quarkaigc.pretrain_engineering import PretrainEngineering

PE = PretrainEngineering()

# yaml
param_conf = "config.yaml"
tconfig = PE.parse_config(param_conf)

# ======================================================
# 模型训练
PE.train(tconfig)
# ======================================================
# PE.model_tensorboard("./encode/merlin/logs")