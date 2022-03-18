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
from collections import OrderedDict

from detectron2.checkpoint import DetectionCheckpointer


def _rename_HRNet_weights(weights):
    # We detect and  rename HRNet weights for DensePose. 1956 and 1716 are values that are
    # common to all HRNet pretrained weights, and should be enough to accurately identify them
    if (
        len(weights["model"].keys()) == 1956
        and len([k for k in weights["model"].keys() if k.startswith("stage")]) == 1716
    ):
        hrnet_weights = OrderedDict()
        for k in weights["model"].keys():
            hrnet_weights["backbone.bottom_up." + str(k)] = weights["model"][k]
        return {"model": hrnet_weights}
    else:
        return weights


class DensePoseCheckpointer(DetectionCheckpointer):
    """
    Same as :class:`DetectionCheckpointer`, but is able to handle HRNet weights
    """

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        super().__init__(model, save_dir, save_to_disk=save_to_disk, **checkpointables)

    def _load_file(self, filename: str) -> object:
        """
        Adding hrnet support
        """
        weights = super()._load_file(filename)
        return _rename_HRNet_weights(weights)
