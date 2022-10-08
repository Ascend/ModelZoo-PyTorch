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


import os
import argparse
import torch
import timm


def pth2onnx(args):
    pth_path = args.input_path
    batch_size = args.batch_size
    model_name = args.model_name
    out_path = args.out_path

    # get size
    if 's3' in model_name:
        size = int(model_name.split('_')[3])
    else:
        size = int(model_name.split('_')[4])
    input_data = torch.randn([batch_size, 3, size, size]).to(torch.float32)
    input_names = ["image"]
    output_names = ["out"]

    # build model
    model = timm.create_model(model_name, checkpoint_path=pth_path)
    model.eval()

    torch.onnx.export(
        model,
        input_data,
        out_path,
        verbose=True,
        opset_version=11,
        input_names=input_names,
        output_names=output_names
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description='SwinTransformer onnx export.')
    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help='input path for pth model')
    parser.add_argument('-o', '--out_path', type=str, required=True,
                        help='save path for output onnx model')
    parser.add_argument('-n', '--model_name', type=str, default='swin_base_patch4_window12_384',
                        help='model name for swintransformer')
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='batch size for output model')
    args = parser.parse_args()
    args.out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    return args


if __name__ == '__main__':
    args = parse_arguments()
    pth2onnx(args)
