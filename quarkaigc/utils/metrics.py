#!/usr/bin/env python
# -*- coding: utf-8 -*-
#@File    :   metrics.py
#@Time    :   2023/10/16 10:32:23
#@Author  :   oliverchen 
#@Moto :   Stay hungary,Stay foolish
#@Version :   1.0
#@Desc    :   None

from __future__ import absolute_import, division, print_function
import torch
from torch import Tensor
from typing_extensions import Literal
from typing import Any, List, Optional, Sequence, Union,Tuple

from torchmetrics import Metric
from torchmetrics.functional.multimodal.clip_score import _clip_score_update, _get_clip_model_and_processor


from transformers import CLIPModel ,CLIPProcessor ,ChineseCLIPModel,ChineseCLIPProcessor
import PIL




class CLIPScore(Metric):
    r"""Calculates `CLIP Score`_ which is a text-to-image similarity metric.

    CLIP Score is a reference free metric that can be used to evaluate the correlation between a generated caption for
    an image and the actual content of the image. It has been found to be highly correlated with human judgement. The
    metric is defined as:

    .. math::
        \text{CLIPScore(I, C)} = max(100 * cos(E_I, E_C), 0)

    which corresponds to the cosine similarity between visual `CLIP`_ embedding :math:`E_i` for an image :math:`i` and
    textual CLIP embedding :math:`E_C` for an caption :math:`C`. The score is bound between 0 and 100 and the closer
    to 100 the better.

    .. note:: Metric is not scriptable

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``images`` (:class:`~torch.Tensor` or list of tensors): tensor with images feed to the feature extractor with. If
        a single tensor it should have shape ``(N, C, H, W)``. If a list of tensors, each tensor should have shape
        ``(C, H, W)``. ``C`` is the number of channels, ``H`` and ``W`` are the height and width of the image.
    - ``text`` (:class:`~str` or :class:`~list` of :class:`~str`): text to compare with the images, one for each image.

    As output of `forward` and `compute` the metric returns the following output

    - ``clip_score`` (:class:`~torch.Tensor`): float scalar tensor with mean CLIP score over samples

    Args:
        model_name_or_path: string indicating the version of the CLIP model to use. Available models are:

            - `"openai/clip-vit-base-patch16"`
            - `"openai/clip-vit-base-patch32"`
            - `"openai/clip-vit-large-patch14-336"`
            - `"openai/clip-vit-large-patch14"`

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If transformers package is not installed or version is lower than 4.10.0

    Example:
        >>> import torch
        >>> from torchmetrics.multimodal.clip_score import CLIPScore
        >>> metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
        >>> score = metric(torch.randint(255, (3, 224, 224), generator=torch.manual_seed(42)), "a photo of a cat")
        >>> score.detach()
        tensor(24.4255)

    """


    def __init__(
        self,
        model_name_or_path: str= "openai/clip-vit-large-patch14" ,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model, self.processor = self._get_clip_model_and_processor(model_name_or_path)
        self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        
    def _get_clip_model_and_processor(self,
		model_name_or_path: str = "openai/clip-vit-large-patch14"
	) -> Tuple[CLIPModel, CLIPProcessor]:
        try:
          
            if "chinese" in model_name_or_path.lower():
                model = ChineseCLIPModel.from_pretrained(model_name_or_path,cache_dir = "./")
                processor = ChineseCLIPProcessor.from_pretrained(model_name_or_path,cache_dir = "./") 
            else:
                model = CLIPModel.from_pretrained(model_name_or_path,cache_dir = "./")
                processor = CLIPProcessor.from_pretrained(model_name_or_path,cache_dir = "./")
            return model,processor 
        except:
            raise ModuleNotFoundError(
            "`clip_score` metric requires `transformers` package be installed."
            " Either install with `pip install transformers>=4.10.0` or `pip install torchmetrics[multimodal]`."
        )
    
    def _clip_score_update(self,
        images: Union[str, List[str]],
        texts: Union[PIL.Image.Image,List[PIL.Image.Image]]
    ) -> Tensor:
   
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        text_embeds=outputs.text_embeds
        image_embeds=outputs.image_embeds
        # cosine similarity between feature vectors
        score = 100 * (text_embeds * image_embeds).sum(axis=-1)
        score = score.mean(0)
        return score, len(texts)
    

    def update(self, images: Union[List[PIL.Image.Image], List[Tensor]], texts: Union[str, List[str]]) -> None:
        """Update CLIP score on a batch of images and text.

        Args:
            images: Either a single [N, C, H, W] tensor or a list of [C, H, W] tensors
            text: Either a single caption or a list of captions

        Raises:
            ValueError:
                If not all images have format [C, H, W]
            ValueError:
                If the number of images and captions do not match

        """
        score, n_samples = self._clip_score_update(images, texts, )
        self.score += score.sum(0)
        self.n_samples += n_samples

    def compute(self) -> Tensor:
        """Compute accumulated clip score."""
        return torch.max(self.score / self.n_samples, torch.zeros_like(self.score))

    