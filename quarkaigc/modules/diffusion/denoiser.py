# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from quarkaigc.modules.util import append_dims, instantiate_from_config


class Denoiser(nn.Module):

    def __init__(self, weighting_config, scaling_config):
        super().__init__()

        self.weighting = instantiate_from_config(weighting_config)
        self.scaling = instantiate_from_config(scaling_config)

    def possibly_quantize_sigma(self, sigma):
        return sigma

    def possibly_quantize_c_noise(self, c_noise):
        return c_noise

    def w(self, sigma):
        return self.weighting(sigma)

    def __call__(self, network, input, sigma, cond):
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        return network(input * c_in, c_noise, cond) * c_out + input * c_skip


class DiscreteDenoiser(Denoiser):

    def __init__(
        self,
        weighting_config,
        scaling_config,
        num_idx,
        discretization_config,
        do_append_zero=False,
        quantize_c_noise=True,
        flip=True,
    ):
        super().__init__(weighting_config, scaling_config)
        sigmas = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=do_append_zero, flip=flip)
        self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise

    def sigma_to_idx(self, sigma):
        dists = sigma - self.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def possibly_quantize_sigma(self, sigma):
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise):
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise


class EDMScaling:

    def __init__(self, sigma_data=0.5):
        self.sigma_data = sigma_data

    def __call__(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2)**0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2)**0.5
        c_noise = 0.25 * sigma.log()
        return c_skip, c_out, c_in, c_noise


class EpsScaling:

    def __call__(self, sigma):
        c_skip = torch.ones_like(sigma, device=sigma.device)
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1.0)**0.5
        c_noise = sigma.clone()
        return c_skip, c_out, c_in, c_noise


class VScaling:

    def __call__(self, sigma):
        c_skip = 1.0 / (sigma**2 + 1.0)
        c_out = -sigma / (sigma**2 + 1.0)**0.5
        c_in = 1.0 / (sigma**2 + 1.0)**0.5
        c_noise = sigma.clone()
        return c_skip, c_out, c_in, c_noise
