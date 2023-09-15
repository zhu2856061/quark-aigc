# -*- coding: utf-8 -*-
# @Time   : 2023/5/10 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import warnings
import pytorch_lightning as pl
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
import torchdata.datapipes as dp


warnings.filterwarnings(action='ignore', category=UserWarning)

#================================================================================
# 一，准备数据
#================================================================================


def input_array(train_config: TrainBuilderConfig, x=None):
    """为了tensorboard 中能够有graph信息，
        lighting 中必须设置 【self.example_input_array】
        将torch 模型转换成onnx文件时也必须要这个固化输入
    """
    for name, fea in train_config.features.items():
        if fea.count > 0:
            x[name] = torch.tensor(x[name], dtype=torch.int64)
        else:
            x[name] = torch.tensor(x[name], dtype=torch.float32)
        if fea.use_len:
            x[name + "_len"] = torch.tensor(x[name + "_len"],
                                            dtype=torch.int64)
    return x


class DataFactoryModule(pl.LightningDataModule):

    def __init__(self, train_config: TrainBuilderConfig,
                 FeatureBuilderObj: FeatureBuilder):
        super().__init__()

        self.tr_files = train_config.tr_files
        self.val_files = train_config.val_files
        self.train_config = train_config
        self.FeatureBuilderObj = FeatureBuilderObj

        self.shuffle_size = train_config.shuffle_size
        self.batch_size = train_config.batch_size
        self.num_workers = train_config.num_workers
        self.pin_memory = train_config.pin_memory

        self.labels = train_config.labels
        self.features = train_config.features

    def setup(self, stage=None):
        if stage == "fit":
            self.tr_ds = self.build_datapipes(self.tr_files)
            self.vl_ds = self.build_datapipes(self.val_files)

        if stage == "validate":
            self.vl_ds = self.build_datapipes(self.val_files)

    def train_dataloader(self):
        """
        当data pipes 与 DataLoader 一起使用时，其中num_workers>0时，
        每个工作进程将具有 DataPipe 对象的不同副本，
        方法：
        1 通常需要独立配置每个副本，以避免从工作线程返回重复数据，
        2 使用两个参数sharding_filter（用于分布式） 和 sharding_round_robin_dispatch（用于多进程）
        """
        ds = self.tr_ds.sharding_filter().sharding_round_robin_dispatch(
            SHARDING_PRIORITIES.MULTIPROCESSING)
        ds = ds.shuffle(buffer_size=self.shuffle_size)

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn if 'gpt'
            not in self.train_config.model_name else self.gpt_collate_fn,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        ds = self.vl_ds.sharding_filter().sharding_round_robin_dispatch(
            SHARDING_PRIORITIES.MULTIPROCESSING)

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn if 'gpt'
            not in self.train_config.model_name else self.gpt_collate_fn,
            pin_memory=self.pin_memory,
        )

    # 读取数据 - 多文件读取
    def build_datapipes(self, files):
        datapipe = dp.iter.FileOpener(files, mode='rt').readlines()
        return datapipe

    # 数据处理逻辑 - 对一个batch进行处理
    def collate_fn(self, batch):
        _batch = {}
        for ba in batch:
            ba = json.loads(ba[1])
            try:
                x, x_len, y = self.FeatureBuilderObj.encoder(ba["x"], ba["y"])
            except KeyError:
                x, x_len, y = self.FeatureBuilderObj.encoder(ba["x"])
            
            for name, fea in self.features.items():
                try:
                    _batch[name].append(x[name])
                    if fea.use_len:
                        _batch[name + "_len"].append(x_len[name])
                except KeyError:
                    _batch[name] = [x[name]]
                    if fea.use_len:
                        _batch[name + "_len"] = [x_len[name]]
            if y is not None:
                for la in self.labels:
                    try:
                        _batch[la.name].append(y[la.name])
                    except KeyError:
                        _batch[la.name] = [y[la.name]]

        # tensor
        _batch = input_array(self.train_config, _batch)
        if self.labels is not None:
            for la in self.labels:
                try:
                    if la.value_type == "float":
                        _batch[la.name] = torch.tensor(_batch[la.name], dtype=torch.float32)
                    elif la.value_type == "int":
                        _batch[la.name] = torch.tensor(_batch[la.name], dtype=torch.int64)
                except KeyError:
                    continue

        return _batch

    # 数据处理逻辑
    # 在生成式，处理中对于句子对的问题【翻译，问答】，
    # 可以把句子拼接起来，中间用<SEP>, 开头用<CLS>
    def gpt_collate_fn(self, batch):
        x_ = []
        y_ = []
        for ba in batch:
            ba = json.loads(ba[1])
            x, _, _ = self.FeatureBuilderObj.encoder(ba["x"])

            data = []
            data.append(4) # <CLS>
            data.extend(x['source'])
            data.append(5) # <SEP>
            data.extend(x['target'])
            data = np.array(data)
            ix = torch.randint(
                len(data) - self.train_config.block_size,
                (self.train_config.batch_size, ))

            for i in ix:
                x_.append(
                    torch.from_numpy(
                        (data[i:i + self.train_config.block_size]).astype(
                            np.int64)))
                y_.append(
                    torch.from_numpy(
                        (data[i + 1:i + 1 +
                              self.train_config.block_size]).astype(np.int64)))
        x = torch.stack(x_)
        y = torch.stack(y_)

        return {'source': x, 'target': y}
