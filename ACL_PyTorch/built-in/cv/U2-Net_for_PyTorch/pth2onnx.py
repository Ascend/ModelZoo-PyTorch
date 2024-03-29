# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import argparse

import torch

sys.path.append('./workspace/U-2-Net')

from model import U2NET


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default="./workspace/U-2-Net/saved_models/u2net/u2net.pth",
                        type=str, help='dir of model')
    parser.add_argument('--out_path', default="./models/u2net.onnx", type=str,
                        help='path of onnx ')
    args = parser.parse_args()
    
    model_dir = args.model_dir
    out_path = args.out_path
    
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    net.eval()

    # build random input
    input_names = ["image"]
    output_names = ["d1", "d2", "d3", "d4", "d5", "d6", "d7"]
    dummy_input = torch.rand(1, 3, 320, 320)
    dynamic_axes = {
        'image': {0: '-1'},
        'd1': {0: '-1'},
        'd2': {0: '-1'},
        'd3': {0: '-1'},
        'd4': {0: '-1'},
        'd5': {0: '-1'},
        'd6': {0: '-1'},
        'd7': {0: '-1'}}

    torch.onnx.export(net, dummy_input, out_path, dynamic_axes=dynamic_axes,
                      verbose=True, input_names=input_names,
                      output_names=output_names, opset_version=11)
