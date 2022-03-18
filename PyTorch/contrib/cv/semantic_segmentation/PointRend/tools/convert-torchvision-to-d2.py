#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle as pkl
import sys
import torch

"""
Usage:
  # download one of the ResNet{18,34,50,101,152} models from torchvision:
  wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O r50.pth
  # run the conversion
  ./convert-torchvision-to-d2.py r50.pth r50.pkl

  # Then, use r50.pkl with the following changes in config:

MODEL:
  WEIGHTS: "/path/to/r50.pkl"
  PIXEL_MEAN: [123.675, 116.280, 103.530]
  PIXEL_STD: [58.395, 57.120, 57.375]
  RESNETS:
    DEPTH: 50
    STRIDE_IN_1X1: False
INPUT:
  FORMAT: "RGB"

  These models typically produce slightly worse results than the
  pre-trained ResNets we use in official configs, which are the
  original ResNet models released by MSRA.
"""

if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")

    newmodel = {}
    for k in list(obj.keys()):
        old_k = k
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
    if obj:
        print("Unconverted keys:", obj.keys())
