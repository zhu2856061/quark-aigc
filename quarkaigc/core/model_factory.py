# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function
from quarknlp.model.Translate import TranslateTrainModule, TranslateInferenceModule

#================================================================================
# 一，准备模型和预估
#================================================================================
class ModelFactoryModule(object):

    def __init__(
        self,
        train_config: TrainBuilderConfig = None,
        FeatureBuilderObj: FeatureBuilder = None,
    ) -> None:
        self.modellib = {
            "translate": TranslateTrainModule,
        }
        self.inferencelib = {
            "translate": TranslateInferenceModule,
        }
        self.FeatureBuilderObj = FeatureBuilderObj
        self.train_config = train_config

    def get_model(self):
        return self.modellib[self.train_config.object](
            train_config=self.train_config,
        )

    def get_inference(self):
        return self.inferencelib[self.train_config.object](
            train_config=self.train_config,
            FeatureBuilderObj=self.FeatureBuilderObj,
        )
