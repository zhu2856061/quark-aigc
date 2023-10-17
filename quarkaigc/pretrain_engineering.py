# -*- coding: utf-8 -*-
# @Time   : 2023/8/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import os
import torch
from loguru import logger
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ModelSummary,
    RichProgressBar,
)

from quarkaigc.utils.utils import (
    instantiate_from_params,
    instantiate_from_one_params,
    instantiate_model_card_from_config
)


class PretrainEngineering(object):

    def __init__(self, encoder_dir: str = "./encode"):
        self.encoder_dir = encoder_dir

    def train(self, config):

        pl.seed_everything(2023, workers=True)

        # 数据
        DataModule = instantiate_from_params(config.lightningModule.data)
        DataModule.setup(stage="fit")
        dataloader = DataModule.val_dataloader()
        example_input = None
        for d in dataloader:
            example_input = d['input_ids']
            break
        # exit()

        # 模型
        LightningModule = instantiate_from_one_params(
            example_input, config.lightningModule.model)

        # 训练
        runtime_conf = config.lightningModule.runtime
        experiment_name = runtime_conf["experiment_name"]
        checkpoint_path = runtime_conf["checkpoint_path"]
        strategy = runtime_conf["strategy"]
        accelerator = runtime_conf["accelerator"]
        devices = runtime_conf["devices"]
        max_epochs = runtime_conf["max_epochs"]
        progress_bar = RichProgressBar(leave=True)
        """ 训练启动器 """
        ckpt_callback = ModelCheckpoint(
            dirpath=os.path.join(self.encoder_dir, experiment_name,
                                 checkpoint_path),
            save_weights_only=True,
            monitor='val/loss',
            save_top_k=1,
            mode='min',
        )

        early_stopping = EarlyStopping(
            monitor='val/loss',
            patience=20,
            mode='min',
        )

        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=os.path.join(self.encoder_dir, experiment_name),
            name='logs',
            log_graph=True,
        )

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            max_epochs=max_epochs,
            callbacks=[
                ModelSummary(max_depth=-1),
                ckpt_callback,
                early_stopping,
                progress_bar,
            ],
            logger=[tb_logger],
        )

        trainer.fit(
            model=LightningModule,
            train_dataloaders=DataModule.train_dataloader(),
            val_dataloaders=DataModule.val_dataloader(),
        )

    def load_model(self, config, ckpt_file):
        """
        导入模型，主要是为了继续训练和预估使用，因此都需要先定义2部分：特征和模型
        """
        # 特征
        self.params = config.lightningModule.data.params
        self.tokenizer = instantiate_from_model_card(self.params)
        self.padding = self.params.get('padding', 'max_length')
        self.truncation = self.params.get('truncation', True)
        self.max_length = self.params.get('max_length', 1024)
        self.return_tensors = self.params.get('return_tensors', 'pt')

        # 模型
        self.LightningModule = instantiate_from_one_params(
            None, config.lightningModule.model)

        # 导入
        checkpoint = torch.load(ckpt_file)
        self.LightningModule.model.load_state_dict({
            k.replace('model.', ''): v
            for k, v in checkpoint['state_dict'].items()
        })
        self.LightningModule.model.eval()
        print(self.LightningModule.model)
        logger.info(
            f">>>>> load {config.lightningModule.model.target} success <<<<<")

    def predict(self, batch):
        """ 加载ckpt文件后，进行预估 其中batch 为多条样本，样本格式为原始格式样本中的x部分 """
        _x = {}
        for ba in batch:
            for name, item in ba.items():
                try:
                    _x[name].append(item['SVA'][0])
                except KeyError:
                    _x[name] = [item['SVA'][0]]

        x = {}
        for name, sentence in _x.items():
            x[name] = self.tokenizer(
                sentence,
                padding=self.padding,
                truncation=self.truncation,
                return_tensors=self.return_tensors,
            )

        pred = self.LightningModule(x)
        return pred

    def model_visual(self, scalar_dir=None, model_path=None):
        from visualdl.server import app
        app.run(
            scalar_dir,
            model=model_path,
            host="127.0.0.1",
            port=8080,
            cache_timeout=60,
            language=None,
            public_path=None,
            api_only=False,
            open_browser=False,
        )

    def model_tensorboard(self, log_dir: str = None):
        os.system(f"tensorboard --logdir={log_dir}")

    def parse_config(self, config: str):
        return OmegaConf.load(config)