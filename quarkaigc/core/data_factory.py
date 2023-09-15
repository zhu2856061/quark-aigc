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
from PIL import Image
from io import BytesIO
import base64
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
import torchdata.datapipes as dp
from torchvision import transforms

from quarkaigc.core.config import BuilderConfig

warnings.filterwarnings(action='ignore', category=UserWarning)

#================================================================================
# 一，准备数据
#================================================================================


class DataFactoryModule(pl.LightningDataModule):

    def __init__(self, config: BuilderConfig):
        super().__init__()

        self.data_config = config.TrainBuilder.Data

        self.tr_files = self.data_config.tr_files
        self.val_files = self.data_config.val_files

        self.shuffle_size = self.data_config.shuffle_size
        self.batch_size = self.data_config.batch_size
        self.num_workers = self.data_config.num_workers
        self.pin_memory = self.data_config.pin_memory

        self.labels = config.FeatureBuilder.Labels
        self.features = config.FeatureBuilder.Features

        self.data_transforms = transforms.Compose([
            transforms.Resize(
                self.data_config.resolution,
                interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.data_config.resolution)
            if self.data_config.center_crop else transforms.RandomCrop(
                self.data_config.resolution),
            transforms.RandomHorizontalFlip() if self.data_config.random_flip
            else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def setup(self, stage=None):
        if stage == "fit":
            self.tr_ds = self.build_datapipes(self.tr_files)
            self.vl_ds = self.build_datapipes(self.val_files)

        if stage == "validate":
            self.vl_ds = self.build_datapipes(self.val_files)

    def train_dataloader(self):
        ds = self.tr_ds.sharding_filter().sharding_round_robin_dispatch(
            SHARDING_PRIORITIES.MULTIPROCESSING)
        ds = ds.shuffle(buffer_size=self.shuffle_size)

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
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
            collate_fn=self.collate_fn,
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
            ba = json.loads(ba[1])['x']
            image = Image.open(BytesIO(base64.b64decode(str.encode(ba['image']['SVA'][0]))))
            image = self.data_transforms(image.convert("RGB"))
            try:
                _batch['image'].append(image)
            except:
                _batch['image'] = [image]

        _batch = {key: torch.stack(value) for key, value in _batch.items()}
        return _batch
