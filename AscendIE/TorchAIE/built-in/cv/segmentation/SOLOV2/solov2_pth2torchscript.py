# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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

import torch
import argparse
import numpy as np
from mmdet.apis import init_detector

input_names = ['input']
output_names = ['seg_preds', 'cate_labels', 'cate_scores']

def pth2torchscript(args, fake_input):
    model = init_detector(args.config, args.pth_path, device='cpu')
    model.forward = model.simple_test
    ts_model = torch.jit.trace(model, fake_input)
    ts_model.save(args.out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='model config')
    parser.add_argument('--out-path', default='./solov2_torchscript.pt', help='onnx output name')
    parser.add_argument('--pth-path', help='model pth path')
    parser.add_argument('--shape', type=int, nargs='+', help='input image size hxw')
    args = parser.parse_args()
    assert len(args.shape) == 2
    fake_input = torch.randn(1, 3, args.shape[0], args.shape[1])
    pth2torchscript(args, fake_input)
