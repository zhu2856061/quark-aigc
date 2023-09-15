# -*- coding: utf-8 -*-
# @Time   : 2023/8/10 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function
from omegaconf import OmegaConf
import dataclasses
from typing import List


@dataclasses.dataclass
class FeatureConfig():
    # 特征配置
    name: str = None
    feature_type: str = None
    value_type: str = 'str'
    transform: str = 'hash'
    min_max: List = None
    bucket: int = 10
    len: int = 1
    default: float | str | int = None

    count: int = 0
    use_len: bool = False
    shared_embed: str = None


@dataclasses.dataclass
class LabelConfig():
    # 目标配置
    name: str = None
    value_type: str = None
    transform: str = None
    metric: str = None
    default: float | str | int = None


@dataclasses.dataclass
class DataConfig():
    # 数据配置
    tr_files: List[str] = None
    val_files: List[str] = None
    shuffle_size: int = 100
    batch_size: int = 16
    num_workers: int = 1
    pin_memory: bool = True

    # 
    resolution: int = 512
    center_crop: bool = False
    random_flip: bool = False


@dataclasses.dataclass
class RuntimeConfig():
    # 训练配置
    experiment_name: str = 'merlin'
    checkpoint_path: str = "./savemodel"
    strategy: str = "auto"
    accelerator: str = "cpu"
    devices: int = 1
    max_epochs: int = 1
    optimizer_name: str = 'adam'
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclasses.dataclass
class ModelConfig():
    # 模型参数
    object: str = None
    model_name: str = None
    embed_size: int = 8
    layer_size: List = None
    bn: bool = True
    dropout: float = 0.5


@dataclasses.dataclass
class FeatureBuilderConfig():
    Features: List[FeatureConfig] = None
    Labels: List[FeatureConfig] = None


@dataclasses.dataclass
class TrainBuilderConfig():
    Data: DataConfig = None
    Model: ModelConfig = None
    Runtime: RuntimeConfig = None


@dataclasses.dataclass
class BuilderConfig():
    FeatureBuilder: FeatureBuilderConfig = None
    TrainBuilder: TrainBuilderConfig = None


def parse_feature_info_config(config_info_path: str):
    info = OmegaConf.load(config_info_path)

    features = []
    if info.features is not None:
        for fea in info.features:
            fe = FeatureConfig(**fea)
            features.append(fe)

    labels = []
    if info.labels is not None:
        for lab in info.labels:
            la = LabelConfig(**lab)
            labels.append(la)

    return FeatureBuilderConfig(features, labels)


def parse_info_config(config_info_path: str):
    info = OmegaConf.load(config_info_path)

    if info.data is not None:
        data = DataConfig(**info.data)
    else:
        data = DataConfig()
    if info.model is not None:
        model = ModelConfig(**info.model)
    else:
        model = ModelConfig()
    if info.runtime is not None:
        runtime = RuntimeConfig(**info.runtime)
    else:
        runtime = RuntimeConfig()

    features = []
    if info.features is not None:
        for fea in info.features:
            fe = FeatureConfig(**fea)
            features.append(fe)

    labels = []
    if info.labels is not None:
        for lab in info.labels:
            la = LabelConfig(**lab)
            labels.append(la)

    feature = FeatureBuilderConfig(features, labels)
    train = TrainBuilderConfig(Data=data, Model=model, Runtime=runtime)

    return BuilderConfig(feature, train)

