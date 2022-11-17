#!/usr/bin/env python3

# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RegNet models."""

import numpy as np
import utils.model.blocks as bk
from utils.model.anynet import AnyNet

def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per stage widths and depths from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    # Generate continuous per-block ws
    ws_cont = np.arange(d) * w_a + w_0
    # Generate quantized per-block ws
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws_all = w_0 * np.power(w_m, ks)
    ws_all = np.round(np.divide(ws_all, q)).astype(int) * q
    # Generate per stage ws and ds (assumes ws_all are sorted)
    ws, ds = np.unique(ws_all, return_counts=True)
    # Compute number of actual stages and total possible stages
    num_stages, total_stages = len(ws), ks.max() + 1
    # Convert numpy arrays to lists and return
    ws, ds, ws_all, ws_cont = (x.tolist() for x in (ws, ds, ws_all, ws_cont))
    return ws, ds, num_stages, total_stages, ws_all, ws_cont


def generate_regnet_full():
    """Generates per stage ws, ds, gs, bs, and ss from RegNet cfg."""
    w_a, w_0, w_m, d = 20.71, 48, 2.65, 27
    ws, ds = generate_regnet(w_a, w_0, w_m, d)[0:2]
    ss = [2 for _ in ws]
    bs = [1.0 for _ in ws]
    gs = [24 for _ in ws]
    ws, bs, gs = bk.adjust_block_compatibility(ws, bs, gs)
    return ws, ds, ss, bs, gs


class RegNet(AnyNet):
    """RegNet model."""

    @staticmethod
    def get_params():
        """Get AnyNet parameters that correspond to the RegNet."""
        ws, ds, ss, bs, gs = generate_regnet_full()
        return {
            "stem_type": "simple_stem_in",
            "stem_w": 32,
            "block_type": "res_bottleneck_block",
            "depths": ds,
            "widths": ws,
            "strides": ss,
            "bot_muls": bs,
            "group_ws": gs,
            "head_w": 0,
            "se_r": 0.25,
            "num_classes": 1000,
        }

    def __init__(self):
        params = RegNet.get_params()
        super(RegNet, self).__init__(params)

    @staticmethod
    def complexity(cx, params=None):
        """Computes model complexity (if you alter the model, make sure to update)."""
        params = RegNet.get_params() if not params else params
        return AnyNet.complexity(cx, params)
