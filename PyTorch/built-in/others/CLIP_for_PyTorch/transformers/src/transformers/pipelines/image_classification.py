# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==========================================================================

from typing import List, Union

from ..file_utils import (
    add_end_docstrings,
    is_tf_available,
    is_torch_available,
    is_vision_available,
    requires_backends,
)
from ..utils import logging
from .base import PIPELINE_INIT_ARGS, Pipeline


if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image

if is_tf_available():
    import tensorflow as tf

    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING

if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING

logger = logging.get_logger(__name__)


@add_end_docstrings(PIPELINE_INIT_ARGS)
class ImageClassificationPipeline(Pipeline):
    """
    Image classification pipeline using any `AutoModelForImageClassification`. This pipeline predicts the class of an
    image.

    This image classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"image-classification"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=image-classification).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        requires_backends(self, "vision")
        self.check_model_type(
            TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING
            if self.framework == "tf"
            else MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING
        )

    def _sanitize_parameters(self, top_k=None):
        postprocess_params = {}
        if top_k is not None:
            postprocess_params["top_k"] = top_k
        return {}, {}, postprocess_params

    def __call__(self, images: Union[str, List[str], "Image.Image", List["Image.Image"]], **kwargs):
        """
        Assign labels to the image(s) passed as inputs.

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images, which must then be passed as a string.
                Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL
                images.
            top_k (`int`, *optional*, defaults to 5):
                The number of top labels that will be returned by the pipeline. If the provided number is higher than
                the number of labels available in the model configuration, it will default to the number of labels.

        Return:
            A dictionary or a list of dictionaries containing result. If the input is a single image, will return a
            dictionary, if the input is a list of several images, will return a list of dictionaries corresponding to
            the images.

            The dictionaries contain the following keys:

            - **label** (`str`) -- The label identified by the model.
            - **score** (`int`) -- The score attributed by the model for that label.
        """
        return super().__call__(images, **kwargs)

    def preprocess(self, image):
        image = load_image(image)
        model_inputs = self.feature_extractor(images=image, return_tensors=self.framework)
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs, top_k=5):
        if top_k > self.model.config.num_labels:
            top_k = self.model.config.num_labels

        if self.framework == "pt":
            probs = model_outputs.logits.softmax(-1)[0]
            scores, ids = probs.topk(top_k)
        elif self.framework == "tf":
            probs = tf.nn.softmax(model_outputs.logits, axis=-1)[0]
            topk = tf.math.top_k(probs, k=top_k)
            scores, ids = topk.values.numpy(), topk.indices.numpy()
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

        scores = scores.tolist()
        ids = ids.tolist()
        return [{"score": score, "label": self.model.config.id2label[_id]} for score, _id in zip(scores, ids)]
