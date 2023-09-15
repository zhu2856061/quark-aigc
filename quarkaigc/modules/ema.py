# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import torch
from copy import deepcopy

"""
Vt=β⋅Vt-1+(1-β)⋅θt
θt: 在第 t 次更新得到的所有参数权重
vt: 第 t 次更新的所有参数移动平均数
β: 权重参数
————————————————
版权声明：本文为CSDN博主「C--G」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_50973728/article/details/129432119
普通的参数权重相当于一直累积更新整个训练过程的梯度，
使用EMA的参数权重相当于使用训练过程梯度的加权平均（刚开始的梯度权值很小）。
由于刚开始训练不稳定，得到的梯度给更小的权值更为合理，所以EMA会有效。
"""

class LitEma(torch.nn.Module):
    """ 指数移动平均
        指数移动平均（Exponential Moving Average）也叫权重移动平均（Weighted Moving Average），是一种给予近期数据更高权重的平均方法。
        在深度学习中，经常会使用EMA（指数移动平均）这个方法对模型的参数做平均，以求提高测试指标并增加模型鲁棒。
    """
    def __init__(self, model: torch.nn.Module, decay=0.9999, use_num_upates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.m_name2s_name = {}
        # 通过register_buffer()登记过的张量：会自动成为模型中的参数，
        # 随着模型移动（gpu/cpu）而移动，但是不会随着梯度进行更新。
        self.register_buffer("decay", torch.tensor(decay, dtype=torch.float32))
        self.register_buffer(
            "num_updates",
            torch.tensor(0, dtype=torch.int)
            if use_num_upates
            else torch.tensor(-1, dtype=torch.int),
        )

        for name, p in model.named_parameters():
            if p.requires_grad:
                # remove as '.'-character is not allowed in buffers
                s_name = name.replace(".", "")
                self.m_name2s_name.update({name: s_name})
                self.register_buffer(s_name, p.clone().detach().data)

        self.collected_params = []

    def reset_num_updates(self):
        del self.num_updates
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.int))

    def forward(self, model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(
                        one_minus_decay * (shadow_params[sname] - m_param[key])
                    )
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

class EMA(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        # perform ema on different device from model if set
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

