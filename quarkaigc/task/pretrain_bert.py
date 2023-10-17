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
from typing import Any

from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric

from quarkaigc.utils.utils import (
    instantiate_from_model_card,
    instantiate_from_config,
    instantiate_from_params_config,
)

warnings.filterwarnings(action='ignore', category=UserWarning)
"""
预训练+bert类模型问题
"""

class DataFactoryModule(pl.LightningDataModule):

    def __init__(self, params) -> None:

        self.params = params

        self.tokenizer = instantiate_from_model_card(params)

        self.padding = self.params.get('padding', 'max_length')
        self.truncation = self.params.get('truncation', False)
        self.max_length = self.params.get('max_length', None)
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
            prefetch_factor=1,
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
            prefetch_factor=1,
        )

    # 读取数据 - 多文件读取
    def build_datapipes(self, files):
        datapipe = dp.iter.FileOpener(files, mode='rt').readlines()
        return datapipe

    # 数据处理逻辑 - 对一个batch进行处理
    def collate_fn(self, batch):
        _x = {}

        for ba in batch:
            ba = json.loads(ba[1])
            for name, item in ba["x"].items():
                try:
                    _x[name].append(item['SVA'][0])
                except KeyError:
                    _x[name] = [item['SVA'][0]]

        x = {}
        y = {}
        tmp = self.tokenizer(
            _x["text"],
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
        )
        y["class"] = tmp["input_ids"].detach().clone()

        # bert 训练的核心就在于这里输入部分mask操作
        # 进行随机mask
        input_ids = tmp["input_ids"]
        rand = torch.rand(input_ids.shape)
        mask_arr = (rand<0.15)*(input_ids!=0)*(input_ids!=1)*(input_ids!=2)*(input_ids!=3)
        for i in range(input_ids.shape[0]):
            selection = torch.flatten(mask_arr[i].nonzero()).tolist()
            input_ids[i,selection] = 4
        tmp["input_ids"] = input_ids
        x["text"] = tmp

        # y["mask"] = mask_arr

        return x, y


class ModelFactoryModule(torch.nn.Module):

    def __init__(self, network_conf) -> None:
        super().__init__()
        # 基座
        self.base = instantiate_from_model_card(network_conf['base'])
        # ======================================
        # 冻结 基座 - 定制化冻结
        for param in self.base.parameters():
            param.requires_grad = True

    def forward(self, x):
        out = self.base(**x['text']) # ['last_hidden_state', 'pooler_output']

        return out


class TaskFactoryModule(pl.LightningModule):

    def __init__(self, example_input, params, *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.params = params
        self.network_conf = self.params['network_config']
        self.loss_conf = self.params['loss_config']
        self.metric_conf = self.params['metric_config']
        self.optimizer_conf = self.params.get(
            'optimizer_config', {
                'target': 'torch.optim.AdamW',
                'params': {
                    'lr': 1e-4,
                    'weight_decay': 1e-4
                }
            })
        self.scheduler_conf = self.params.get('scheduler_config', None)

        self.save_hyperparameters(self.params, ignore=["example_input"])

        # 0. define model input example;
        # self.example_input_array = example_input,

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

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        output = self(batch[0])
        tgt = batch[1]["class"]

        # 这里就是需要转成两维的preds [n, vocab_size] 与 真实的logits [n]
        a, b, c = output[0].shape
        output = output[0].view(a*b, -1)
        tgt = tgt.view(-1)
        loss = self.loss(output, tgt)
        preds = torch.argmax(output, dim=-1)

        return {"loss": loss, "preds": preds, "y": tgt}

    # 【train】==================================================================
    def on_train_epoch_start(self):
        self.train_losses.reset()
        self.train_metrics.reset()

    def training_step(self, batch, batch_idx):
        outputs = self._shared_step(batch)

        # 收集
        self.train_losses.update(outputs["loss"])
        self.train_metrics.update(outputs["preds"], outputs["y"])

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

        return outputs["loss"]

    def on_validation_epoch_start(self):
        self.eval_metrics.reset()
        self.eval_losses.reset()

    def validation_step(self, batch, batch_idx):
        outputs = self._shared_step(batch)
        # 收集
        self.eval_losses.update(outputs["loss"])
        self.eval_metrics.update(outputs["preds"], outputs["y"])

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
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        # 4. define optimizer
        optim_groups = [{
            'params':
            decay_params,
            'weight_decay':
            self.optimizer_conf['params'].pop('weight_decay'),
        }, {
            'params': nodecay_params,
            'weight_decay': 0.0,
        }]

        fused = True if self.device.type == "cuda" else False
        self.optimizer_conf['params'].update({"fused": fused})
        optimizer = instantiate_from_params_config(optim_groups,
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
