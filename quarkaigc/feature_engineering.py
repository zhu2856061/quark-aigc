# -*- coding: utf-8 -*-
# @Time   : 2023/8/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

from quarknlp.core.data_utils import FeatureBuilder
from quarknlp.core.config import FeatureBuilderConfig, parse_feature_info_config


class FeatureEngineering(object):

    def __init__(self, encoder_dir: str = "./encode"):
        self.encoder_dir = encoder_dir

    def parse_config(self, config: str):
        """
        对 feature info 文件进行读取，并转换成dataclass
        """
        return parse_feature_info_config(config)