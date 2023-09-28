# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

from torch import nn
from typing import Dict, Any


class FeedForward(nn.Module):
    def __init__(self, n_layers, dropout, bias, *args: Any, **kwargs: Any):
        super().__init__()
        self.dense = nn.ModuleList()
        for i in range(len(n_layers) - 1):
            if i == len(n_layers) - 2:
                m = nn.ModuleDict(dict(
                    fc=nn.Linear(n_layers[i], n_layers[i+1], bias=bias),
                    gelu=nn.GELU(),
                ))
                self.dense.append(m)
            else:
                m = nn.ModuleDict(dict(
                    fc=nn.Linear(n_layers[i], n_layers[i+1], bias=bias),
                    gelu=nn.GELU(),
                    dropout=nn.Dropout(dropout),
                ))
                self.dense.append(m)

    def forward(self, x):
        for i, l in enumerate(self.dense):
            if i == len(self.dense) - 1:
                x = l.fc(x)
                x = l.gelu(x)
            else:
                x = l.fc(x)
                x = l.gelu(x)
                x = l.dropout(x)

        return x
