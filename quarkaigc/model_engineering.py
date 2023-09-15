# -*- coding: utf-8 -*-
# @Time   : 2023/8/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import os
from visualdl import LogWriter
from quarknlp.core.config import parse_train_info_config
from quarknlp.core.config import TrainBuilderConfig
from quarknlp.core.data_utils import FeatureBuilder

from quarknlp.core.train_factory import TrainerFactoryModule


class ModelEngineering(object):

    def __init__(self, encoder_dir: str = "./encode"):
        self.encoder_dir = encoder_dir

    def train(self, train_config: TrainBuilderConfig,
              FeatureBuilderObj: FeatureBuilder):

        # 记录每个特征的长度，其中若长度为0，则不用embed层
        train_config.features = FeatureBuilderObj.feature2dict
        # 训练
        TrainerModule = TrainerFactoryModule(
            train_config,
            self.encoder_dir,
        )
        TrainerModule.trainer()

    def parse_config(self, config: str):
        """
        对 train info 文件进行读取，并转换成dataclass
        """
        return parse_train_info_config(config)