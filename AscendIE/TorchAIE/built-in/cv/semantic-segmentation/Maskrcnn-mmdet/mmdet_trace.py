# Copyright 2023 Huawei Technologies Co., Ltd
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

import sys
import os
import numpy as np
import torch
import argparse
sys.path.append("./mmdetection")

from mmdetection.mmdet.core import generate_inputs_and_wrap_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="./mmdetection/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth")
    parser.add_argument("--config", type=str, default="./mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py")
    parser.add_argument("--save_path", type=str, default="./mmdet_torch.ts")
    args = parser.parse_args()

    torch.ops.load_library("./mmdet_ops/build/libmmdet_ops.so")

    input_shape = (1, 3, 1216, 1216)
    input_img = './mmdetection/tests/data/color.jpg'
    normalize_cfg = {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}
    input_config = {
        'input_shape': input_shape,
        'input_path': input_img,
        'normalize_cfg': normalize_cfg
    }

    model, tensor_data = generate_inputs_and_wrap_model(args.config, args.ckpt, input_config)
    model.cpu().eval()

    dummy_input = tensor_data
    ts_model = torch.jit.trace(model, example_inputs=dummy_input)
    print("Finish converting")

    ts_model.save(args.save_path)


if __name__ == "__main__":
    main()
