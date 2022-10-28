# Copyright 2021 Huawei Technologies Co., Ltd
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
sys.path.append('./ctpn.pytorch')
import config
from ctpn.ctpn import CTPN_Model
import argparse
import torch



def ctpn_pth2onnx(args):
    """[change pth to onnx]

    Args:
        args ([argparse]): [change model parameters]
    """
    weights = args.pth_path
    model = CTPN_Model()
    model.load_state_dict(torch.load(weights, map_location='cpu')['model_state_dict'])
    model.eval()
    input_names = ['image']
    output_names = ['class', 'regression']
    for i in range(config.center_len):
        h, w = config.center_list[i][0], config.center_list[i][1]
        dummy_input = torch.randn(1, 3, h, w)
        output_file = '{}/ctpn_{}x{}.onnx'.format(args.onnx_path, h, w)
        torch.onnx.export(model, dummy_input, output_file, input_names = input_names, 
        output_names = output_names, opset_version=11, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ctpn pth to onnx') # change model parameters
    parser.add_argument('-p', '--pth_path', default='./ctpn.pytorch/weights/ctpn.pth',
                        type=str, help='pth model path')
    parser.add_argument('-o', '--onnx_path', default='./',
                        type=str, help='onnx model path')
    args = parser.parse_args()
    ctpn_pth2onnx(args)