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
from torchvision import transforms
from typing import Dict, Any

from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric

from quarkaigc.utils.utils import (
    instantiate_model_card_from_config,
    instantiate_from_config,
    instantiate_from_one_params,
    instantiate_from_params_config
)
from tqdm import tqdm
import csv
csv.field_size_limit(100000000) #for parse,csv error

from PIL import Image
from io import BytesIO
import base64

warnings.filterwarnings(action='ignore', category=UserWarning)


class DataFactoryModule(pl.LightningDataModule):

    def __init__(self, config) -> None:

        self.tokenizer = instantiate_model_card_from_config(config)

        self.params = config.get("params", dict())

        self.padding = self.params.get('padding', 'max_length')
        self.truncation = self.params.get('truncation', True)
        # self.max_length = self.params.get('max_length', 1024)
        self.return_tensors = self.params.get('return_tensors', 'pt')

        self.tr_files = self.params.get('tr_files')
        self.val_files = self.params.get('val_files')
        self.shuffle_size = self.params.get('shuffle_size', 1024)
        self.batch_size = self.params.get('batch_size', 16)
        self.num_workers = self.params.get('num_workers', 1)
        self.pin_memory = self.params.get('pin_memory', False)

        self.resolution=self.params.get('resolution',768)
        self.center_crop=self.params.get('center_crop', False)
        self.random_flip=self.params.get("random_flip",True)

        self.image_transform = transforms.Compose(
                        [
                            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                            transforms.CenterCrop(self.resolution) if self.center_crop else transforms.RandomCrop(self.resolution),
                            transforms.RandomHorizontalFlip() if self.random_flip else transforms.Lambda(lambda x: x),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5]),
                        ]
                    )



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
            collate_fn=self.train_collate_fn,
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
            collate_fn=self.val_collate_fn,
            pin_memory=self.pin_memory,
        )
    #解析csv文件
    def row_processor(self,row):
        return {"text": row[4], "image": Image.open(BytesIO(base64.b64decode(str.encode(row[3])))),
                'hash_id':int(row[0]),'sku_id':row[1],'url':row[2],'width':int(row[5]),
                'height':int(row[6]),'clip_score':float(row[7]),'aesth':float(row[8])}
    
    # 读取数据 - 多文件读取
    def build_datapipes(self, files):
        datapipe = dp.iter.FileOpener(files, mode='rt').parse_csv(delimiter=",")
        datapipe=datapipe.map(self.row_processor)
        return datapipe

    # 数据处理逻辑 - 对一个batch进行处理
    def train_collate_fn(self, batch):
        pixel_values = torch.stack([self.image_transform(example['image'].convert("RGB")) for example in batch])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = self.tokenizer(
            [example['text'] for example in batch],  padding="max_length", 
            truncation=self.truncation, return_tensors=self.return_tensors
        ).input_ids
        _batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return _batch
    
    def val_collate_fn(self, batch):
        pixel_values = torch.stack([self.image_transform(example['image'].convert("RGB")) for example in batch])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = self.tokenizer(
            [example['text'] for example in batch], padding="max_length", 
            truncation=self.truncation, return_tensors=self.return_tensors
        ).input_ids
        batch_size= input_ids.shape[0]
        neg_input_ids= self.tokenizer(
            ['']*batch_size, padding="max_length", 
            truncation=self.truncation, return_tensors=self.return_tensors
        ).input_ids
        _batch = {
            "input_ids": input_ids,
            'neg_input_ids':neg_input_ids,
            "pixel_values": pixel_values,
            'text':[example['text'] for example in batch],
            'image':[example['image'].convert("RGB") for example in batch]
        }
        return _batch
    




class ModelFactoryModule(torch.nn.Module):

    def __init__(self, network_conf) -> None:
        super().__init__()
        # 基座
        self.params = network_conf.get("params", dict())
        self.free_vae=self.params.get("free_vae",True)
        self.free_text_enocoder=self.params.get("free_text_encoder",True)
        self.clip_skip=self.params.get("clip_skip",0)
        self.mixed_precision=self.params.get("mixed_precision",None)
        self.noise_offset=self.params.get("noise_offset",True)
        base_model = instantiate_model_card_from_config(network_conf['base'])
        self.weight_dtype = torch.float32

        

        self.net = torch.nn.ModuleDict(dict(
            vae=base_model.vae,
            text_encoder=base_model.text_encoder,
            unet=base_model.unet,
        ))
        #一些模型参数
        self.vae_scale_factor = 2 ** (len(self.net.vae.config.block_out_channels) - 1)
        #转化双精度
        if not self.mixed_precision:
            if self.mixed_precision=='fp16':
               self.weight_dtype = torch.float16
            elif self.mixed_precision=='bf16':
                self.weight_dtype = torch.bfloat16
        self.net.vae.to(dtype=self.weight_dtype)
        self.net.text_encoder.to(dtype=self.weight_dtype)
        # 冻结 基座 - 定制化冻结
        if self.free_vae:
            self.net.vae.requires_grad_(False)
        #text encoder 可能会放开上面的层
        if self.free_text_enocoder:
            for name, param in list(self.net.text_encoder.named_parameters()): 
                # print('This layer will be frozen: {}'.format(name)) 
                # for param in self.base.parameters():
                param.requires_grad = False
        self.noise_scheduler=base_model.scheduler

    
    def decode_latents(self, latents):
        
        latents = 1 / self.net.vae.config.scaling_factor * latents
        image = self.net.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float()
        return image
    
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype,generator, device, latents=None):
        
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)

        if latents is None:
            latents = torch.randn(shape, generator=generator, device=torch.device("cpu"), dtype=dtype,layout=torch.strided).to(device)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.noise_scheduler.init_noise_sigma
        return latents
    
    def encode_text_ids(self,input_ids):
        if not self.clip_skip:
            encoder_hidden_states = self.net.text_encoder(input_ids)[0]
        else:
            encoder_hidden_states=self.net.text_encoder(
                    input_ids, output_hidden_states=True
                )
            encoder_hidden_states = encoder_hidden_states[-1][-(self.clip_skip + 1)]
            # We also need to apply the final LayerNorm here to not mess with the
            # representations. The `last_hidden_states` that we typically use for
            # obtaining the final prompt representations passes through the LayerNorm
            # layer.
            encoder_hidden_states = self.net.text_encoder.text_model.final_layer_norm(encoder_hidden_states)
        return encoder_hidden_states
    
    def forward(self, x: Dict):
        # Convert images to latent space
        latents = self.net.vae.encode(x["pixel_values"].to(self.weight_dtype)).latent_dist.sample()
        latents = latents * self.net.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if self.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )

        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.encode_text_ids(x["input_ids"])


        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # Predict the noise residual and compute loss
        model_pred = self.net.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        return target,model_pred

    
    



class TaskFactoryModule(pl.LightningModule):

    def __init__(self, example_input, params, *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.params = params
        self.network_conf = self.params['network_config']
        self.loss_conf = self.params['loss_config']
        # self.metric_conf = self.params['metric_config']
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

        # 1. define model
        self.model = ModelFactoryModule(self.network_conf)

        # 2. define loss
        self.train_losses = MeanMetric()
        self.eval_losses = MeanMetric()
        self.loss = instantiate_from_config(self.loss_conf)

        # # 3. define metric
        # self.train_metrics = {}
        # self.eval_metrics = {}
        # for metric in self.metric_conf:
        #     self.eval_metrics[metric['name']] = instantiate_from_config(metric)
        # self.eval_metrics = MetricCollection(self.eval_metrics)

    def forward(self, x):
        return self.model(x)
    
    def generate(self,x,height=512,width=512,num_inference_steps=20,
                 num_images_per_prompt=1,
                 guidance_scale: float = 7.5):
        do_classifier_free_guidance = guidance_scale > 1.0
        # 0. Default height and width to unet
        height = height or self.model.net.unet.config.sample_size * self.model.vae_scale_factor
        width = width or self.mode.net.unet.config.sample_size * self.model.vae_scale_factor
        # 3. Encode input prompt
        prompt_embeds =self.model.encode_text_ids(x['input_ids'])
        batch_size=prompt_embeds.shape[0]
        if do_classifier_free_guidance:
            neg_prompt_embeds =self.model.encode_text_ids(x['neg_input_ids'])
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds])

        
        device=prompt_embeds.device

         # 4. Prepare timesteps
        self.model.noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.model.noise_scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.model.net.unet.config.in_channels
        generator=torch.manual_seed(2023)#how to get lighting modolue seed?
        latents = self.model.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            generator,
            device,
        )
        # 7. Denoising loop
        for t in tqdm(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.model.noise_scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.model.net.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.model.noise_scheduler.step(noise_pred, t, latents).prev_sample
        image = self.model.decode_latents(latents)

        image=image.numpy()
        return image

    def _shared_step(self, batch):
        y_true,y_pred = self(batch)
        loss = self.loss(y_true, y_pred)
        
        return {"loss": loss, "preds": y_pred, "y": y_true}

    # 【train】==================================================================
    def on_train_epoch_start(self):
        self.train_losses.reset()

    def training_step(self, batch, batch_idx):
        outputs = self._shared_step(batch)

        # 收集
        self.train_losses.update(outputs["loss"])

        # 评估loss
        _loss = self.train_losses.compute()
        self.log(f"train_step/loss",
                 _loss,
                 prog_bar=True,
                 sync_dist=True,
                 rank_zero_only=True)


        return outputs["loss"]

    def on_validation_epoch_start(self):

        # for _,metric in self.eval_metrics.items():
        #     metric.reset()
        self.eval_losses.reset()

    def validation_step(self, batch, batch_idx):
        outputs = self._shared_step(batch)
        # images=self.generate(batch)
        # 收集
        self.eval_losses.update(outputs["loss"])
        # for name,metric in self.eval_metrics.items():
        #     if name=='clip_score':
        #         metric.update(images,batch['text'])
        #     elif name=='fid':
        #         metric.update(batch['image'],real=True)
        #         metric.update(torch.tensor(images),real=False)
                


    def on_validation_epoch_end(self):
        # 评估loss
        _loss = self.eval_losses.compute()
        self.log(f"val/loss",
                 _loss,
                 prog_bar=True,
                 sync_dist=True,
                 rank_zero_only=True)
        # # 评估metric
        
        # for m, v in self.eval_metrics.items():
        #     v.compute()
        #     self.log(f"val/{m}",
        #              v,
        #              prog_bar=True,
        #              sync_dist=True,
        #              rank_zero_only=True)

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
