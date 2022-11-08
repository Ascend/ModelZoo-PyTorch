# Copyright 2022 Huawei Technologies Co., Ltd
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

sys.path.append('./VideoPose3D')
from common.model import TemporalModel
from collections import OrderedDict
import argparse
import torch


def proc_nodes_module(checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if "module." in k:
            name = k.replace("module.", "")
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def vp3d_path2onnx(args):
    num_joints = 17
    joints_dim = 2
    filter_widths = [3, 3, 3, 3, 3]

    model_pos = TemporalModel(num_joints, joints_dim, num_joints, 
                              filter_widths=filter_widths, causal=False, 
                              dropout=0.25, channels=1024, dense=False)
    dummy_input = torch.randn(2, 6115, num_joints, joints_dim)
    chk_filename = args.model
    print(f'Loading checkpoint {chk_filename}')
    checkpoint = torch.load(chk_filename, map_location='cpu')
    checkpoint = proc_nodes_module(checkpoint['model_pos'])
    model_pos.load_state_dict(checkpoint)

    output_file = args.onnx
    input_names = ['2d_poses']
    output_names = ['3d_preds']

    model_pos.eval()
    torch.onnx.export(model_pos, dummy_input, output_file, input_names=input_names, 
                      output_names=output_names, opset_version=11, verbose=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="vp3d to onnx")
    parser.add_argument('-m', '--model', default='./checkpoint/model_best.bin',
                        type=str, metavar='PATH', help="path to model")
    parser.add_argument('-o', '--onnx', default='vp3d.onnx')

    args = parser.parse_args()
    vp3d_path2onnx(args)
