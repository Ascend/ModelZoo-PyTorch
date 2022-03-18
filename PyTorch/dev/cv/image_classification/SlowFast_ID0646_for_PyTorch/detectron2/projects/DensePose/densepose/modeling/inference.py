#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================
#
# Copyright (c) Facebook, Inc. and its affiliates.
from dataclasses import fields
from typing import Any, List
import torch

from detectron2.structures import Instances


def densepose_inference(densepose_predictor_output: Any, detections: List[Instances]):
    """
    Splits DensePose predictor outputs into chunks, each chunk corresponds to
    detections on one image. Predictor output chunks are stored in `pred_densepose`
    attribute of the corresponding `Instances` object.

    Args:
        densepose_predictor_output: a dataclass instance (can be of different types,
            depending on predictor used for inference). Each field can be `None`
            (if the corresponding output was not inferred) or a tensor of size
            [N, ...], where N = N_1 + N_2 + .. + N_k is a total number of
            detections on all images, N_1 is the number of detections on image 1,
            N_2 is the number of detections on image 2, etc.
        detections: a list of objects of type `Instance`, k-th object corresponds
            to detections on k-th image.
    """
    k = 0
    for detection_i in detections:
        if densepose_predictor_output is None:
            # don't add `pred_densepose` attribute
            continue
        n_i = len(detection_i)
        PredictorOutput = type(densepose_predictor_output)
        output_i_dict = {}
        # we assume here that `densepose_predictor_output` is a dataclass object
        for field in fields(densepose_predictor_output):
            field_value = getattr(densepose_predictor_output, field.name)
            # slice tensors
            if isinstance(field_value, torch.Tensor):
                output_i_dict[field.name] = field_value[k : k + n_i]
            # leave others as is
            else:
                output_i_dict[field.name] = field_value
        detection_i.pred_densepose = PredictorOutput(**output_i_dict)
        k += n_i
