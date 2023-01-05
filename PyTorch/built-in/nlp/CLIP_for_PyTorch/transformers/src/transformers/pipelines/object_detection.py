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

from typing import Any, Dict, List, Union

from ..file_utils import add_end_docstrings, is_torch_available, is_vision_available, requires_backends
from ..utils import logging
from .base import PIPELINE_INIT_ARGS, Pipeline


if is_vision_available():
    from ..image_utils import load_image


if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_OBJECT_DETECTION_MAPPING

logger = logging.get_logger(__name__)


Prediction = Dict[str, Any]
Predictions = List[Prediction]


@add_end_docstrings(PIPELINE_INIT_ARGS)
class ObjectDetectionPipeline(Pipeline):
    """
    Object detection pipeline using any `AutoModelForObjectDetection`. This pipeline predicts bounding boxes of objects
    and their classes.

    This object detection pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"object-detection"`.

    See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=object-detection).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        requires_backends(self, "vision")
        self.check_model_type(MODEL_FOR_OBJECT_DETECTION_MAPPING)

    def _sanitize_parameters(self, **kwargs):
        postprocess_kwargs = {}
        if "threshold" in kwargs:
            postprocess_kwargs["threshold"] = kwargs["threshold"]
        return {}, {}, postprocess_kwargs

    def __call__(self, *args, **kwargs) -> Union[Predictions, List[Prediction]]:
        """
        Detect objects (bounding boxes & classes) in the image(s) passed as inputs.

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing an HTTP(S) link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images. Images in a batch must all be in the
                same format: all as HTTP(S) links, all as local paths, or all as PIL images.
            threshold (`float`, *optional*, defaults to 0.9):
                The probability necessary to make a prediction.

        Return:
            A list of dictionaries or a list of list of dictionaries containing the result. If the input is a single
            image, will return a list of dictionaries, if the input is a list of several images, will return a list of
            list of dictionaries corresponding to each image.

            The dictionaries contain the following keys:

            - **label** (`str`) -- The class label identified by the model.
            - **score** (`float`) -- The score attributed by the model for that label.
            - **box** (`List[Dict[str, int]]`) -- The bounding box of detected object in image's original size.
        """

        return super().__call__(*args, **kwargs)

    def preprocess(self, image):
        image = load_image(image)
        target_size = torch.IntTensor([[image.height, image.width]])
        inputs = self.feature_extractor(images=[image], return_tensors="pt")
        inputs["target_size"] = target_size
        return inputs

    def _forward(self, model_inputs):
        target_size = model_inputs.pop("target_size")
        outputs = self.model(**model_inputs)
        model_outputs = outputs.__class__({"target_size": target_size, **outputs})
        return model_outputs

    def postprocess(self, model_outputs, threshold=0.9):
        target_size = model_outputs["target_size"]
        raw_annotations = self.feature_extractor.post_process(model_outputs, target_size)
        raw_annotation = raw_annotations[0]
        keep = raw_annotation["scores"] > threshold
        scores = raw_annotation["scores"][keep]
        labels = raw_annotation["labels"][keep]
        boxes = raw_annotation["boxes"][keep]

        raw_annotation["scores"] = scores.tolist()
        raw_annotation["labels"] = [self.model.config.id2label[label.item()] for label in labels]
        raw_annotation["boxes"] = [self._get_bounding_box(box) for box in boxes]

        # {"scores": [...], ...} --> [{"score":x, ...}, ...]
        keys = ["score", "label", "box"]
        annotation = [
            dict(zip(keys, vals))
            for vals in zip(raw_annotation["scores"], raw_annotation["labels"], raw_annotation["boxes"])
        ]

        return annotation

    def _get_bounding_box(self, box: "torch.Tensor") -> Dict[str, int]:
        """
        Turns list [xmin, xmax, ymin, ymax] into dict { "xmin": xmin, ... }

        Args:
            box (`torch.Tensor`): Tensor containing the coordinates in corners format.

        Returns:
            bbox (`Dict[str, int]`): Dict containing the coordinates in corners format.
        """
        if self.framework != "pt":
            raise ValueError("The ObjectDetectionPipeline is only available in PyTorch.")
        xmin, ymin, xmax, ymax = box.int().tolist()
        bbox = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        }
        return bbox
