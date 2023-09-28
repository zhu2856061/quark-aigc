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
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
import torchdata.datapipes as dp
from typing import Dict, Any

from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric

from quarkaigc.task.utils import (
    instantiate_model_card_from_config,
    instantiate_from_config,
    instantiate_optimizer_from_config,
)

warnings.filterwarnings(action='ignore', category=UserWarning)


class DataFactoryModule(pl.LightningDataModule):

    def __init__(self, config) -> None:

        self.tokenizer = instantiate_model_card_from_config(config)

        self.params = config.get("params", dict())

        self.padding = self.params.get('padding', 'max_length')
        self.truncation = self.params.get('truncation', True)
        self.max_length = self.params.get('max_length', 1024)
        self.return_tensors = self.params.get('return_tensors', 'pt')

        self.tr_files = self.params.get('tr_files')
        self.val_files = self.params.get('val_files')
        self.shuffle_size = self.params.get('shuffle_size', 1024)
        self.batch_size = self.params.get('batch_size', 16)
        self.num_workers = self.params.get('num_workers', 1)
        self.pin_memory = self.params.get('pin_memory', False)

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
        _y = []
        for ba in batch:
            ba = json.loads(ba[1])
            for name, item in ba["x"].items():
                try:
                    _batch[name].append(item['SVA'][0])
                except KeyError:
                    _batch[name] = [item['SVA'][0]]
            _y.append(ba["y"]['FV'])

        res = {}
        for name, sentence in _batch.items():
            res[name] = self.tokenizer(
                sentence,
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_length,
                return_tensors=self.return_tensors,
            )
        res['y'] = torch.tensor(_y, dtype=torch.int64)
        return res


class ModelFactoryModule(torch.nn.Module):

    def __init__(self, network_conf) -> None:
        super().__init__()
        # 基座
        base_model = instantiate_model_card_from_config(network_conf['base'])

        self.net = torch.nn.ModuleDict(dict(
            vae=base_model.vae,
            text_encoder=base_model.text_encoder,
            text_encoder_2=base_model.text_encoder_2,
            unet=base_model.unet,
        ))

    def forward(self, x: Dict):
        text_out = self.net.text_encoder_2(x['text'])
        img_out = self.net.vae(x['img'])
        # 组合

        return out


class TaskFactoryModule(pl.LightningModule):

    def __init__(self, config, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.config = config
        self.network_conf = self.config.network_config
        self.loss_conf = self.config.loss_config
        self.metric_conf = self.config.metric_config
        self.optimizer_conf = self.config.optimizer_config
        self.scheduler_conf = self.config.get('scheduler_config', None)

        self.save_hyperparameters(self.config)

        # 0. define model input example
        # self.example_input_array = input_array(self.config)

        # 1. define model
        self.model = ModelFactoryModule(self.network_conf)

        # 2. define loss
        self.train_losses = MeanMetric()
        self.eval_losses = MeanMetric()
        self.loss = instantiate_from_config(self.loss_conf)

        # 3. define metric
        self.train_metrics = {}
        self.eval_metrics = {}
        for metric in self.metric_conf:
            self.train_metrics[metric['name']] = instantiate_from_config(
                metric)
            self.eval_metrics[metric['name']] = instantiate_from_config(metric)
        self.train_metrics = MetricCollection(self.train_metrics)
        self.eval_metrics = MetricCollection(self.eval_metrics)

    def forward(self, x: Dict):
        return self.model(x['text'])

    def _shared_step(self, batch):
        output = self(batch)
        tgt = batch["y"]
        loss = self.loss(output, tgt)
        preds = torch.argmax(output, dim=-1)
        ret = {"loss": loss, "preds": preds, "y": tgt}
        return ret

    # 【train】==================================================================
    def on_train_epoch_start(self):
        self.train_losses.reset()
        self.train_metrics.reset()

    def training_step(self, batch, batch_idx):
        ret = self._shared_step(batch)

        # 收集
        self.train_losses.update(ret["loss"])
        self.train_metrics.update(ret["preds"], ret["y"])

        # 评估loss
        _loss = self.train_losses.compute()
        self.log(f"train_step/loss",
                 _loss,
                 prog_bar=True,
                 sync_dist=True,
                 rank_zero_only=True)

        # 评估acc
        _met = self.train_metrics.compute()
        for m, v in _met.items():
            self.log(f"train_step/{m}",
                     v,
                     prog_bar=True,
                     sync_dist=True,
                     rank_zero_only=True)

    def on_validation_epoch_start(self):
        self.eval_metrics.reset()
        self.eval_losses.reset()

    def validation_step(self, batch, batch_idx):
        ret = self._shared_step(batch)
        # 收集
        self.eval_losses.update(ret["loss"])
        self.eval_metrics.update(ret["preds"], ret["y"])

    def on_validation_epoch_end(self):
        # 评估loss
        _loss = self.eval_losses.compute()
        self.log(f"val/loss",
                 _loss,
                 prog_bar=True,
                 sync_dist=True,
                 rank_zero_only=True)
        # 评估metric
        _met = self.eval_metrics.compute()
        for m, v in _met.items():
            self.log(f"val/{m}",
                     v,
                     prog_bar=True,
                     sync_dist=True,
                     rank_zero_only=True)

    def configure_optimizers(self):
        if self.global_rank == 0:
            print(self.model)
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param = [p for pn, p in param_dict.items() if p.requires_grad]
        # 4. define optimizer
        optim_groups = [{
            'params': param,
            'weight_decay':
            self.optimizer_conf['params'].pop('weight_decay'),
        }]

        fused = True if self.device.type == "cuda" else False
        self.optimizer_conf['params'].update({"fused": fused})
        optimizer = instantiate_optimizer_from_config(optim_groups,
                                                      self.optimizer_conf)

        # 4.5. define scheduler
        if self.scheduler_conf is not None:
            scheduler = instantiate_from_config(self.scheduler_conf)
            print("Setting up LambdaLR scheduler...")
            scheduler = {
                "scheduler": LambdaLR(optimizer, scheduler.schedule),
                "interval": "step",
                "frequency": 1,
            }
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}

        return {'optimizer': optimizer}
