
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
from CGAN import generator
import torch
import torch.onnx
import sys
from collections import OrderedDict
import argparse


def parse_args():
    desc = "Pytorch implementation of CGAN collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--input_dim', type=int, default=62, help="The input_dim")
    parser.add_argument('--output_dim', type=int, default=3, help="The output_dim")
    parser.add_argument('--input_size', type=int, default=28, help="The image size of MNIST")
    parser.add_argument('--class_num', type=int, default=10, help="The num of classes of MNIST")
    parser.add_argument('--pth_path', type=str, default='CGAN_G.pth', help='pth model path')
    parser.add_argument('--onnx_path', type=str, default="CGAN.onnx", help='onnx model path')
    return parser.parse_args()


def proc_nodes_module(checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if "module." in k:
            name = k.replace("module.", "")
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

def pth2onnx():
    args = parse_args()
    net = generator(input_dim=args.input_dim, output_dim=args.output_dim,
                    input_size=args.input_size, class_num=args.class_num)
    model = net 
    checkpoint = torch.load(args.pth_path, map_location='cpu')
    checkpoint = proc_nodes_module(checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()
    input_names = ["image"]
    output_names = ["output1"]
    #dynamic_axes = {'image': {0: '-1'}, 'output1': {0: '-1'}}
    dummy_input1 = torch.randn(100, 62)
    dummy_input2 = torch.randn(100, 10)    
    torch.onnx.export(model, (dummy_input1,dummy_input2), args.onnx_path, input_names=input_names,
                      output_names=output_names, opset_version=11, verbose=True)
    print("this model could generete pictures, specifically digits")
    print('onnx export done.')


if __name__ == "__main__":
    pth2onnx()