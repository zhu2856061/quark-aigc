# -*- coding: utf-8 -*-
# @Time   : 2023/10/19 10:24
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import os
import warnings
from loguru import logger
import pytorch_lightning as pl
import torch
from torch import nn

from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
import torchdata.datapipes as dp
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.aggregation import MeanMetric
from torchmetrics import MetricCollection
import numpy as np

from omegaconf import OmegaConf
from pytorch_lightning import loggers as pl_loggers

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ModelSummary,
    RichProgressBar,
)

from data_utils import (
    instantiate_from_config,
    instantiate_from_params_config,
)


torch.multiprocessing.set_sharing_strategy("file_system")
warnings.filterwarnings(action="ignore", category=UserWarning)


class ModuleEngine(object):
    def __init__(self, encoder_dir="./encode"):
        os.makedirs(encoder_dir, exist_ok=True)
        self.config = None
        self.encoder_dir = encoder_dir

        self.model_ckpt_warm_up = False
        self.model_onnx_warm_up = False

    # ===== 训练启动架 =====
    def set_config(self, config):
        self.config = config

    def set_collate_fn_hook(self, collate_fn_hook):
        self.collate_fn_hook = collate_fn_hook

    def set_model_hook(self, model_hook):
        self.model_hook = model_hook

    # =====================

    def train(self):
        """整个训练过程包含4部分:
        特征，用于确定特征的处理方法和网络中输入层处理
        数据，用于对数据处理并输入到网络中
        模型，设计网络
        训练，启动训练，对网络中的参数进行训练
        """
        pl.seed_everything(2023, workers=True)

        # 数据
        DataModule = DataFactoryModule(
            self.collate_fn_hook,
            self.config.lightningModule.data,
        )
        DataModule.setup(stage="fit")
        dataloader = DataModule.train_dataloader()
        example_input = None
        for d in dataloader:
            example_input = d
            break

        # 模型
        LightningModule = TaskFactoryModule(
            example_input,
            self.model_hook,
            self.config.lightningModule.model,
        )

        # 训练
        runtime_conf = self.config.lightningModule.runtime
        experiment_name = runtime_conf["experiment_name"]
        checkpoint_path = runtime_conf["checkpoint_path"]
        strategy = runtime_conf["strategy"]
        accelerator = runtime_conf["accelerator"]
        devices = runtime_conf["devices"]
        max_epochs = runtime_conf["max_epochs"]
        ckpt_early_stopping_monitor = runtime_conf["monitor"]
        ckpt_early_stopping_mode = runtime_conf["mode"]
        progress_bar = RichProgressBar(leave=True)
        """ 训练启动器 """
        ckpt_callback = ModelCheckpoint(
            dirpath=os.path.join(self.encoder_dir, experiment_name, checkpoint_path),
            save_weights_only=True,
            monitor=ckpt_early_stopping_monitor,
            save_top_k=1,
            mode=ckpt_early_stopping_mode,
        )
        early_stopping = EarlyStopping(
            monitor=ckpt_early_stopping_monitor,
            patience=20,
            mode=ckpt_early_stopping_mode,
        )
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=os.path.join(self.encoder_dir, experiment_name),
            name="logs",
            log_graph=False,
        )
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            max_epochs=max_epochs,
            callbacks=[
                ModelSummary(max_depth=-1),
                progress_bar,
                ckpt_callback,
                early_stopping,
            ],
            logger=[tb_logger],
        )
        trainer.fit(
            model=LightningModule,
            train_dataloaders=DataModule.train_dataloader(),
            val_dataloaders=DataModule.val_dataloader(),
        )
        logger.info(f">>>>> train success <<<<<")

    def load_model_2_onnx(self, ckpt_file, onnx_file):
        """导入ckpt 模型转换成 onnx格式，能够加快训练"""
        # 数据
        DataModule = DataFactoryModule(
            self.collate_fn_hook,
            self.config.lightningModule.data,
        )
        DataModule.setup(stage="fit")
        dataloader = DataModule.val_dataloader()
        example_input = None
        for d in dataloader:
            example_input = d[0]
            break

        # 模型
        self._load_model_ckpt(ckpt_file)
        # 这里需要特征留意： input_name 必须采用 example_input 的，这样才能保持一致
        input_name = [name for name, _ in example_input.items()]
        dynamic_axes = {name: {0: "batch"} for name in input_name}
        output_name = ["output"]
        torch.onnx.export(
            self.LightningModule.model,
            {"x": example_input},
            f=onnx_file,
            input_names=input_name,
            output_names=output_name,
            verbose=True,
            dynamic_axes=dynamic_axes,
        )
        logger.info(f">>>>> load_model_2_onnx success <<<<<")

    def _load_model_ckpt(self, ckpt_file: str):
        """
        导入模型，主要是为了继续训练和预估使用，因此都需要先定义2部分：特征和模型
        """
        # 模型
        self.LightningModule = TaskFactoryModule(
            None,
            self.model_hook,
            self.config.lightningModule.model,
        )

        # 导入
        checkpoint = torch.load(ckpt_file)
        self.LightningModule.model.load_state_dict(
            {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
        )
        self.LightningModule.model.eval()
        self.model_ckpt_warm_up = True

        print(self.LightningModule.model)
        logger.info(f">>>>> load model ckpt success <<<<<")

    def _load_model_onnx(self, onnx_file: str):
        """导入 onnx格式的模型文件，用于后面的预估"""
        # 记录每个特征的长度，其中若长度为0，则不用embed层
        import onnxruntime

        self.session = onnxruntime.InferenceSession(
            onnx_file,
            providers=[
                "TensorrtExecutionProvider",
                "CPUExecutionProvider",
                #'CUDAExecutionProvider',
            ],
        )
        # 输入
        self.onnx_input_name = self.session.get_inputs()
        self.model_onnx_warm_up = True

        logger.info(f">>>>> load model onnx success <<<<<")

    def predict_ckpt(self, FE, batch, ckpt_file: str):
        """加载ckpt文件后，进行预估 其中batch 为多条样本，样本格式为原始格式样本中的x部分"""
        if not self.model_ckpt_warm_up:
            self._load_model_ckpt(ckpt_file)
        _batch = {}
        for ba in batch:
            x, y = FE.encoder(ba, return_tensors=True)
            for k, v in x.items():
                try:
                    _batch[k].append(v)
                except:
                    _batch[k] = [v]

        _batch = {k: torch.stack(v) for k, v in _batch.items()}

        pred = self.LightningModule(_batch)
        return pred

    def predict_onnx(self, FE, batch, onnx_file: str):
        """加载onnx文件后，进行预估 其中batch 为多条样本，样本格式为原始格式样本中的x部分"""
        if not self.model_onnx_warm_up:
            self._load_model_onnx(onnx_file)
        _batch = {}
        for ba in batch:
            x, y = FE.encoder(ba, return_tensors=True)
            for k, v in x.items():
                try:
                    _batch[k].append(v)
                except:
                    _batch[k] = [v]

        _batch = {
            input_name.name: np.stack(_batch[input_name.name])
            for input_name in self.onnx_input_name
        }
        pred = self.session.run(None, _batch)
        return pred

    def model_visual(self, model_path):
        """对onnx文件进行可视化展示，能够清晰的看到网络的结构"""
        from visualdl.server import app

        app.run(
            model=model_path,
            host="127.0.0.1",
            port=8080,
            cache_timeout=60,
            language=None,
            public_path=None,
            api_only=False,
            open_browser=False,
        )
        logger.info(f">>>>> model_visual success <<<<<")

    def model_tensorboard(self, log_dir=None):
        """对训练后的各种指标进行可视化展示，用于分析"""
        if log_dir is None:
            log_dir = os.path.join(
                self.encoder_dir,
                self.config.lightningModule.runtime.experiment_name,
                "logs",
            )
        os.system(f"tensorboard --logdir={log_dir}")
        logger.info(f">>>>> model_tensorboard success <<<<<")

    def parse_config(self, config):
        self.config = OmegaConf.load(config)
        return self.config


class DataFactoryModule(pl.LightningDataModule):
    def __init__(self, collate_fn_hook, params):
        self.collate_fn = collate_fn_hook
        self.params = params
        self.return_tensors = self.params.get("return_tensors", True)
        self.tr_files = self.params.get("tr_files", None)
        self.val_files = self.params.get("val_files", None)
        self.shuffle_size = self.params.get("shuffle_size", 1024)
        self.batch_size = self.params.get("batch_size", 16)
        self.num_workers = self.params.get("num_workers", 1)
        self.pin_memory = self.params.get("pin_memory", False)
        self.need_val = True
        if self.val_files is None:
            self.need_val = False

    def setup(self, stage="fit"):
        if stage == "fit":
            self.tr_ds = self.build_datapipes(self.tr_files)
            if self.need_val:
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
            SHARDING_PRIORITIES.MULTIPROCESSING
        )
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
        if self.need_val:
            ds = self.vl_ds.sharding_filter().sharding_round_robin_dispatch(
                SHARDING_PRIORITIES.MULTIPROCESSING
            )

            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=self.pin_memory,
            )
        else:
            return None

    # 读取数据 - 多文件读取
    def build_datapipes(self, files):
        if os.path.isfile(files):
            files = [files]
        else:
            files = [
                os.path.join(files, f)
                for f in os.listdir(files)
                if os.path.isfile(os.path.join(files, f))
            ]
        logger.info(f"=== datapipe files: {files} ===")
        datapipe = dp.iter.FileOpener(files, mode="rt").readlines()
        return datapipe


class TaskFactoryModule(pl.LightningModule):
    def __init__(
        self,
        example_input,
        model_hook,
        params,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.params = params
        self.loss_conf = self.params["loss_config"]
        self.metric_conf = self.params["metric_config"]

        self.optimizer_conf = self.params.get(
            "optimizer_config",
            {
                "target": "torch.optim.AdamW",
                "params": {"lr": 1e-4, "weight_decay": 1e-4},
            },
        )
        self.scheduler_conf = self.params.get("scheduler_config", None)

        self.save_hyperparameters(
            self.params,
            ignore=["example_input", "model_hook"],
        )

        # 0. define model input example; 特别留意这个括号的用意， 这里是由于为了生成图内部处理机制造成
        # pytorch_lightning/utilities/model_summary/model_summary.py 277行，查看代码
        if example_input is not None:
            # self.example_input_array = {"x": example_input[0]}
            self.example_input_array = (example_input[0],)
        # 1. define model
        self.model = model_hook
        # self.model = torch.compile(self.model)
        self._init_weights()

        # 2. define loss
        self.losses = MeanMetric()
        self.loss = instantiate_from_config(self.loss_conf)

        # 3. define metric
        self.pred_mean = MeanMetric()
        self.y_mean = MeanMetric()
        self.train_metrics = {}
        self.eval_metrics = {}
        for metric in self.metric_conf:
            self.train_metrics[metric["name"]] = instantiate_from_config(metric)
            self.eval_metrics[metric["name"]] = instantiate_from_config(metric)
        self.train_metrics = MetricCollection(self.train_metrics)
        self.eval_metrics = MetricCollection(self.eval_metrics)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        logit = self(batch[0])
        if len(batch) >= 2:
            tmp = self.model._shared_step(self.loss, logit, batch[1])
        else:
            tmp = self.model._shared_step(self.loss, logit, None)
        return tmp

    # 【train】==================================================================
    def on_train_epoch_start(self):
        self.losses.reset()
        self.pred_mean.reset()
        self.y_mean.reset()
        self.train_metrics.reset()

    def training_step(self, batch, batch_idx):
        outputs = self._shared_step(batch)
        # 收集
        self.losses.update(outputs["loss"])
        self.pred_mean.update(outputs["preds"])
        self.y_mean.update(outputs["y"])
        self.train_metrics.update(outputs["preds"], outputs["y"])
        # 评估loss
        _loss = self.losses.compute()
        self.log(
            f"train_step/loss",
            _loss,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        _y_mean = self.y_mean.compute()
        self.log(
            f"train_step/y_avg",
            _y_mean,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        _pred_mean = self.pred_mean.compute()
        self.log(
            f"train_step/pred_avg",
            _pred_mean,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        # 评估metric
        _met = self.train_metrics.compute()
        for m, v in _met.items():
            self.log(
                f"train_step/{m}", v, prog_bar=True, sync_dist=True, rank_zero_only=True
            )
        return outputs["loss"]

    # 【val】==================================================================
    def on_validation_epoch_start(self):
        self.eval_metrics.reset()
        self.pred_mean.reset()
        self.y_mean.reset()
        self.losses.reset()

    def validation_step(self, batch, batch_idx):
        outputs = self._shared_step(batch)
        # 收集
        self.losses.update(outputs["loss"])
        self.eval_metrics.update(outputs["preds"], outputs["y"])
        self.pred_mean.update(outputs["preds"])
        self.y_mean.update(outputs["y"])

    def on_validation_epoch_end(self):
        # 评估loss
        _loss = self.losses.compute()
        self.log(f"val/loss", _loss, prog_bar=True, sync_dist=True, rank_zero_only=True)
        _y_mean = self.y_mean.compute()
        self.log(
            f"val/y_avg", _y_mean, prog_bar=True, sync_dist=True, rank_zero_only=True
        )
        _pred_mean = self.pred_mean.compute()
        self.log(
            f"val/pred_avg",
            _pred_mean,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        # 评估metric
        _met = self.eval_metrics.compute()
        for m, v in _met.items():
            self.log(f"val/{m}", v, prog_bar=True, sync_dist=True, rank_zero_only=True)

    def configure_optimizers(self):
        if self.global_rank == 0:
            print(self.model)
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
        optim_groups = [
            {
                "params": decay_params,
                "weight_decay": self.optimizer_conf["params"].pop("weight_decay"),
            },
            {
                "params": nodecay_params,
                "weight_decay": 0.0,
            },
        ]

        fused = True if self.device.type == "cuda" else False
        self.optimizer_conf["params"].update({"fused": fused})
        optimizer = instantiate_from_params_config(optim_groups, self.optimizer_conf)

        # 4.5. define scheduler
        if self.scheduler_conf is not None:
            scheduler = instantiate_from_config(self.scheduler_conf)
            print("Setting up LambdaLR scheduler...")
            scheduler = {
                "scheduler": LambdaLR(optimizer, scheduler.schedule),
                "interval": "step",
                "frequency": 1,
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return {"optimizer": optimizer}

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
