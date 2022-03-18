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
"""
python3.7 OpenPose_pth2onnx.py
--checkpoint-path ./weights/checkpoint_iter_370000.pth
--output-name ./output/human-pose-estimation.onnx
"""
import argparse
import os
import sys
import torch
sys.path.append("./lightweight-human-pose-estimation.pytorch")
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state


def convert_to_onnx(network, output_name):
    net_input = torch.randn(1, 3, 368, 640)
    input_names = ['data']
    output_names = ['stage_0_output_1_heatmaps', 'stage_0_output_0_pafs',
                    'stage_1_output_1_heatmaps', 'stage_1_output_0_pafs']
    dynamic_axes = {'data': {0: '-1'}, 'stage_0_output_1_heatmaps': {0: '-1'}, 'stage_0_output_0_pafs': {0: '-1'},
                    'stage_1_output_1_heatmaps': {0: '-1'}, 'stage_1_output_0_pafs': {0: '-1'}}
    torch.onnx.export(network, net_input, output_name, opset_version=11, verbose=True,
                      input_names=input_names, dynamic_axes=dynamic_axes, output_names=output_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='./weights/checkpoint_iter_370000.pth',
                        help='path to the checkpoint')
    parser.add_argument('--output_name', type=str, default='./output/human-pose-estimation.onnx',
                        help='name of output model in ONNX format')
    args = parser.parse_args()

    # mkdir
    dir1, file1 = os.path.split(args.checkpoint_path)
    dir2, file2 = os.path.split(args.output_name)
    if not os.path.exists(dir1):
        os.mkdir(dir1)
    else:
        print(dir1, "already exist")
    if not os.path.exists(dir2):
        os.mkdir(dir2)
    else:
        print(dir2, "already exist")

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))
    load_state(net, checkpoint)

    convert_to_onnx(net, args.output_name)
