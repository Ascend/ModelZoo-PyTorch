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

from dataclasses import make_dataclass
from functools import lru_cache
from typing import Any, Optional
import torch


@lru_cache(maxsize=None)
def decorate_cse_predictor_output_class_with_confidences(BasePredictorOutput: type) -> type:
    """
    Create a new output class from an existing one by adding new attributes
    related to confidence estimation:
    - coarse_segm_confidence (tensor)

    Details on confidence estimation parameters can be found in:
    N. Neverova, D. Novotny, A. Vedaldi "Correlated Uncertainty for Learning
        Dense Correspondences from Noisy Labels", p. 918--926, in Proc. NIPS 2019
    A. Sanakoyeu et al., Transferring Dense Pose to Proximal Animal Classes, CVPR 2020

    The new class inherits the provided `BasePredictorOutput` class,
    it's name is composed of the name of the provided class and
    "WithConfidences" suffix.

    Args:
        BasePredictorOutput (type): output type to which confidence data
            is to be added, assumed to be a dataclass
    Return:
        New dataclass derived from the provided one that has attributes
        for confidence estimation
    """

    PredictorOutput = make_dataclass(
        BasePredictorOutput.__name__ + "WithConfidences",
        fields=[  # pyre-ignore[6]
            ("coarse_segm_confidence", Optional[torch.Tensor], None),
        ],
        bases=(BasePredictorOutput,),
    )

    # add possibility to index PredictorOutput

    def slice_if_not_none(data, item):
        if data is None:
            return None
        if isinstance(item, int):
            return data[item].unsqueeze(0)
        return data[item]

    def PredictorOutput_getitem(self, item):
        PredictorOutput = type(self)
        base_predictor_output_sliced = super(PredictorOutput, self).__getitem__(item)
        return PredictorOutput(
            **base_predictor_output_sliced.__dict__,
            coarse_segm_confidence=slice_if_not_none(self.coarse_segm_confidence, item),
        )

    PredictorOutput.__getitem__ = PredictorOutput_getitem

    def PredictorOutput_to(self, device: torch.device):
        """
        Transfers all tensors to the given device
        """
        PredictorOutput = type(self)
        base_predictor_output_to = super(PredictorOutput, self).to(device)  # pyre-ignore[16]

        def to_device_if_tensor(var: Any):
            if isinstance(var, torch.Tensor):
                return var.to(device)
            return var

        return PredictorOutput(
            **base_predictor_output_to.__dict__,
            coarse_segm_confidence=to_device_if_tensor(self.coarse_segm_confidence),
        )

    PredictorOutput.to = PredictorOutput_to
    return PredictorOutput
