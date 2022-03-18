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
import os
import torch
import numpy as np
import os
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
    parser.add_argument('--save_path', type=str, default="data", help='processed data path')
    return parser.parse_args()


# fixed noise & condition
def prep_preocess(args):
    sample_num = args.class_num**2
    z_dim = args.input_dim
    sample_z_ = torch.zeros((sample_num, z_dim))
    for i in range(args.class_num):
        sample_z_[i * args.class_num] = torch.rand(1,z_dim)
        for j in range(1, args.class_num):
            sample_z_[i * args.class_num + j] = sample_z_[i * args.class_num]

    if not os.path.exists(os.path.join(args.save_path)):
        os.makedirs(os.path.join(args.save_path))

    temp = torch.zeros((args.class_num, 1))
    for i in range(args.class_num):
        temp[i, 0] = i

    temp_y = torch.zeros((sample_num, 1))
    for i in range(args.class_num):
        temp_y[i * args.class_num: (i + 1) * args.class_num] = temp

    sample_y_ = torch.zeros((sample_num, args.class_num)).scatter_(1, temp_y.type(torch.LongTensor), 1)

    input = torch.cat([sample_z_, sample_y_], 1)
    input = np.array(input).astype(np.float32)
    input.tofile(os.path.join(args.save_path, 'input' + '.bin'))
if __name__ == "__main__":
    args = parse_args()
    prep_preocess(args)
    print("data preprocessed stored in",args.save_path)