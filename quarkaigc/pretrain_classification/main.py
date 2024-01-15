# -*- coding: utf-8 -*-
# @Time   : 2023/10/19 10:24
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function
import sys

sys.path.append("../alib/")
from module_utils import ModuleEngine
from data_utils import (
    instantiate_from_config,
    instantiate_from_model_card,
)

import argparse
import torch
import json
from tqdm import tqdm
from omegaconf import OmegaConf

import warnings

warnings.filterwarnings(action="ignore", category=UserWarning)

from transformers import AutoTokenizer
from peft import LoraConfig

# 测试你的tokenizer
# tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
# text = ["Replace me by any text you'd like.", "Hello, my dog is cute"]
# encoded_input = tokenizer(text, return_tensors='pt', padding='max_length')
# print(encoded_input['input_ids'].shape)
# exit()

"""
预训练+分类 问题
"""
ModuleE = ModuleEngine()


# 定义模型
class ModelFactoryModule(torch.nn.Module):
    def __init__(self, network_conf) -> None:
        super().__init__()
        # 基座
        self.base = instantiate_from_model_card(network_conf["base"])

        # ======================================
        # 冻结 基座 - 定制化冻结
        for param in self.base.parameters():
            param.requires_grad = False
        #
        lora_config = LoraConfig(
            target_modules=["query", "key", "value"], init_lora_weights=False
        )
        self.base.add_adapter(lora_config)
        print(self.base)
        # 任务模型
        self.task = instantiate_from_config(network_conf["task"])

    def forward(self, x):
        out = self.base(**x["text"])  # ['last_hidden_state', 'pooler_output']

        out = self.task(out["last_hidden_state"][:, 0, :])

        return out

    def _shared_step(self, loss_fn, logit, y):
        y = y.view(-1)
        loss = loss_fn(logit, y)
        logit = torch.argmax(logit, dim=-1)
        return {"loss": loss, "preds": logit, "y": y}


# 定义数据处理逻辑 - 对一个batch进行处理
def collate_fn_hook(batch):
    _x = {}
    _y = []
    for ba in batch:
        ba = json.loads(ba[1])
        for name, item in ba["x"].items():
            try:
                _x[name].append(item["SVA"][0])
            except KeyError:
                _x[name] = [item["SVA"][0]]
        _y.append(ba["y"]["class"]["IVA"][0])

    x = {}
    for name, sentence in _x.items():
        x[name] = tokenizer(
            sentence,
            padding=data_config.padding,
            truncation=data_config.truncation,
            return_tensors=data_config.return_tensors,
            max_length=data_config.max_length,
        )

    y = torch.tensor(_y, dtype=torch.int64)
    return x, y


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description="CTR")
    parser.add_argument("--task", default="train", help="task")

    parser.add_argument("--config", default="config.yaml", help="config doc")

    parser.add_argument(
        "--ckpt_file",
        default="./encode/merlin/savemodel/epoch=0-step=1018.ckpt",
        help="ckpt_file",
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    p = parser.parse_args()
    config = OmegaConf.load(p.config)
    data_config = config.lightningModule.data
    tokenizer = instantiate_from_model_card(data_config)

    if p.task == "train":
        # 导入词表
        ModuleE.set_config(config)
        model_hook = ModelFactoryModule(
            config.lightningModule.model.network_config,
        )
        # 初始hook
        ModuleE.set_collate_fn_hook(collate_fn_hook)
        ModuleE.set_model_hook(model_hook)
        # ======================================================
        # 模型训练
        ModuleE.train()

