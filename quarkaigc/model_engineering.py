# -*- coding: utf-8 -*-
# @Time   : 2023/8/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import os
from quarkaigc.core.train_factory import TrainerFactoryModule
from quarkaigc.core.data_factory import DataFactoryModule
from quarkaigc.core.config import BuilderConfig, parse_info_config

class ModelEngineering(object):

    def __init__(self, encoder_dir: str = "./encode"):
        self.encoder_dir = encoder_dir

    def train(self, builder_config: BuilderConfig):

        # data
        DataModule = DataFactoryModule(builder_config)
        DataModule.setup(stage="fit")
        ds = DataModule.train_dataloader()
        for d in ds:
            for k, v in d.items():
                print(k, "-->" ,v.shape)
            break
        exit()

        # model
        ModelModule = ModelFactoryModule(train_config, writer).get_model()

        # train
        TrainerFactoryModule(train_config,self.encoder_dir).fit(
            ModelModule,
            train_dataloaders=DataModule.train_dataloader(),
            val_dataloaders=DataModule.val_dataloader(),
        )

    def parse_config(self, config: str):
        """
        对 train info 文件进行读取，并转换成dataclass
        """
        return parse_info_config(config)