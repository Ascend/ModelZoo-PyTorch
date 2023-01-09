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

import argparse
import sys

import torch
import numpy as np
from tqdm import tqdm

sys.path.append(r"./FLAVR")
from FLAVR.model.FLAVR_arch import UNet_3D_3D

def parse_args():
    parser = argparse.ArgumentParser(description='convert pth to onnx')
    parser.add_argument('--model', type=str, default='unet_18')
    parser.add_argument('--input_file', type=str, required=True,
                        help='path to pth model')
    parser.add_argument('--output_file', type=str, required=True,
                        help='path to save onnx model')
    parser.add_argument('--image_size', type=str, default=224,
                        help='path to save onnx model')
    parser.add_argument('--opset_version', type=int, default=13,
                        help='onnx opset version')
    parser.add_argument('--nbr_frame', type=int, default=4)
    parser.add_argument('--joinType', choices=['concat', 'add', 'none'], default='concat')
    parser.add_argument('--n_outputs', type=int, default=3,
                        help='For Kx FLAVR, use n_outputs k-1')
    args = parser.parse_args()
    return args

def pth2onnx(model, input_data, output_file, opset_version):
    model.eval()
    input_names = ['input_0', 'input_1', 'input_2', 'input_3']
    output_names = ['output_0', 'output_1', 'output_2']
    dynamic_axes = {
        'input_0':{0:'-1'},
        'input_1':{0:'-1'},
        'input_2':{0:'-1'},
        'input_3':{0:'-1'},
        'output_0':{0:'-1'},
        'output_1':{0:'-1'},
        'output_2':{0:'-1'}
    }
    with torch.no_grad():
        torch.onnx.export(
            model.module,
            input_data,
            output_file,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_params=True,
            keep_initializers_as_inputs=False,
            verbose=False,
            opset_version=opset_version
        )
    print('Successfully exported onnx model:{}'.format(output_file))

if __name__ == "__main__":
    args = parse_args()
    
    model = UNet_3D_3D(
        args.model, 
        n_inputs=args.nbr_frame,
        n_outputs=args.n_outputs,
        joinType=args.joinType
        )
    model = torch.nn.DataParallel(model).to('cpu')
    model_dict = model.state_dict()
    model.load_state_dict(torch.load(args.input_file, map_location='cpu')['state_dict'], strict=True)
    
    input_data = [torch.randn(1, 3, args.image_size, args.image_size)] * 4
    pth2onnx(model, input_data, args.output_file, args.opset_version)