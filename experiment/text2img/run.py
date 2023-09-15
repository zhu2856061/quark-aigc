# -*- coding: utf-8 -*-
# @Time   : 2023/8/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function
import sys

sys.path.append("../..")
import os
import json
from quarkaigc.feature_engineering import FeatureEngineering
from quarkaigc.model_engineering import ModelEngineering

FE = FeatureEngineering()
ME = ModelEngineering()

# yaml
conf = "config.yaml"
fconfig = FE.parse_config(conf)
tconfig = ME.parse_config(conf)

# ======================================================
# 模型训练
ME.train(tconfig)
