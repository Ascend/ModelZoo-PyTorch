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
import torch

from densepose.structures import DensePoseChartPredictorOutput, DensePoseTransformData


def densepose_chart_predictor_output_hflip(
    densepose_predictor_output: DensePoseChartPredictorOutput,
    transform_data: DensePoseTransformData,
) -> DensePoseChartPredictorOutput:
    """
    Change  to take into account a Horizontal flip.
    """
    if len(densepose_predictor_output) > 0:

        PredictorOutput = type(densepose_predictor_output)
        output_dict = {}

        for field in fields(densepose_predictor_output):
            field_value = getattr(densepose_predictor_output, field.name)
            # flip tensors
            if isinstance(field_value, torch.Tensor):
                setattr(densepose_predictor_output, field.name, torch.flip(field_value, [3]))

        densepose_predictor_output = _flip_iuv_semantics_tensor(
            densepose_predictor_output, transform_data
        )
        densepose_predictor_output = _flip_segm_semantics_tensor(
            densepose_predictor_output, transform_data
        )

        for field in fields(densepose_predictor_output):
            output_dict[field.name] = getattr(densepose_predictor_output, field.name)

        return PredictorOutput(**output_dict)
    else:
        return densepose_predictor_output


def _flip_iuv_semantics_tensor(
    densepose_predictor_output: DensePoseChartPredictorOutput,
    dp_transform_data: DensePoseTransformData,
) -> DensePoseChartPredictorOutput:
    point_label_symmetries = dp_transform_data.point_label_symmetries
    uv_symmetries = dp_transform_data.uv_symmetries

    N, C, H, W = densepose_predictor_output.u.shape
    u_loc = (densepose_predictor_output.u[:, 1:, :, :].clamp(0, 1) * 255).long()
    v_loc = (densepose_predictor_output.v[:, 1:, :, :].clamp(0, 1) * 255).long()
    Iindex = torch.arange(C - 1, device=densepose_predictor_output.u.device)[
        None, :, None, None
    ].expand(N, C - 1, H, W)
    densepose_predictor_output.u[:, 1:, :, :] = uv_symmetries["U_transforms"][Iindex, v_loc, u_loc]
    densepose_predictor_output.v[:, 1:, :, :] = uv_symmetries["V_transforms"][Iindex, v_loc, u_loc]

    for el in ["fine_segm", "u", "v"]:
        densepose_predictor_output.__dict__[el] = densepose_predictor_output.__dict__[el][
            :, point_label_symmetries, :, :
        ]
    return densepose_predictor_output


def _flip_segm_semantics_tensor(
    densepose_predictor_output: DensePoseChartPredictorOutput, dp_transform_data
):
    if densepose_predictor_output.coarse_segm.shape[1] > 2:
        densepose_predictor_output.coarse_segm = densepose_predictor_output.coarse_segm[
            :, dp_transform_data.mask_label_symmetries, :, :
        ]
    return densepose_predictor_output
