
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse

import megengine as mge
import numpy as np
from megengine import jit

from build import build_and_load


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo Dump")
    parser.add_argument("-n", "--name", type=str, default="yolox-s", help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--dump_path", default="model.mge", help="path to save the dumped model"
    )
    return parser


def dump_static_graph(model, graph_name="model.mge"):
    model.eval()
    model.head.decode_in_inference = False

    data = mge.Tensor(np.random.random((1, 3, 640, 640)))

    @jit.trace(capture_as_const=True)
    def pred_func(data):
        outputs = model(data)
        return outputs

    pred_func(data)
    pred_func.dump(
        graph_name,
        arg_names=["data"],
        optimize_for_inference=True,
        enable_fuse_conv_bias_nonlinearity=True,
    )


def main(args):
    model = build_and_load(args.ckpt, name=args.name)
    dump_static_graph(model, args.dump_path)


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
