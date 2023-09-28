# -*- coding: utf-8 -*-
# @Time   : 2023/8/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import os
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ModelSummary,
    RichProgressBar,
)

from quarkaigc.task.utils import (
    instantiate_data_from_config,
    instantiate_task_from_config
)

class PretrainEngineering(object):

    def __init__(self, encoder_dir: str = "./encode"):
        self.encoder_dir = encoder_dir

    def train(self, config):

        pl.seed_everything(2023, workers=True)

        # 数据
        DataModule = instantiate_data_from_config(config)
        DataModule.setup(stage="fit")

        # 模型
        LightningModule = instantiate_task_from_config(config)

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

    def model_tensorboard(self, log_dir: str = None):
        os.system(f"tensorboard --logdir={log_dir}")

    def parse_config(self, config: str):
        return OmegaConf.load(config)