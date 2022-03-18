# encoding=utf-8
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

from __future__ import absolute_import, division, print_function
import argparse
import os
import torch
import parse


def pt2onnx(ar):
    data, model, _ = parse.load_data_model(ar)
    input = tuple(data[0].values())
    # 跟踪一个批次的运算
    model.eval()
    torch.onnx.export(model, input, ar.onnx_path,
                      input_names=['input_ids', 'attention_mask',
                                   'token_type_ids'],  # add input name
                      output_names=['output'],
                      export_params=True, verbose=False, training=False, opset_version=11)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_dir", type=str, default='./albert_pytorch/outputs/SST-2/',
                        help="dir of pth, load args.bin and model.bin")
    parser.add_argument("--onnx_dir", type=str, default='./outputs/',
                        help="dir of onnx, gen onnx name via bs")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size.")
    ar = parser.parse_args()

    ar.pth_arg_path = ar.pth_dir + "training_args.bin"
    ar.data_type = 'dev'
    ar.data_path = './albert_pytorch/dataset/SST-2'
    ar.model_name = "albert_bs{}".format(ar.batch_size)
    ar.onnx_path = "{}/{}.onnx".format(ar.onnx_dir, ar.model_name)

    if not os.path.exists(ar.onnx_dir):
        os.makedirs(ar.onnx_dir)
    pt2onnx(ar)


if __name__ == "__main__":
    main()
