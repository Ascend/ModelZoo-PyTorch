# Copyright 2021 Huawei Technologies Co., Ltd
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

from collections import OrderedDict
import torch
import torch.nn as nn
from CGAN import generator
import argparse
import os
import numpy as np
import utils


def proc_nodes_module(checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if "module." in k:
            name = k.replace("module.", "")
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict
    

def parse_args():
    desc = "Pytorch implementation of CGAN collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--input_dim', type=int, default=62, help="The input_dim")
    parser.add_argument('--output_dim', type=int, default=3, help="The output_dim")
    parser.add_argument('--input_size', type=int, default=28, help="The image size of MNIST")
    parser.add_argument('--class_num', type=int, default=10, help="The num of classes of MNIST")
    parser.add_argument('--pth_path', type=str, default='CGAN_G.pth', help='pth model path')
    parser.add_argument('--onnx_path', type=str, default="CGAN.onnx", help='onnx model path')
    parser.add_argument('--save_path', type=str, default='demo', help="the generated image path")
    return parser.parse_args()
    

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

    return sample_z_, sample_y_

        
def main():
    args = parse_args()
    # Build model
    local_device = torch.device("npu:0")
    torch.npu.set_device(local_device)
    print("using npu :{}".format(local_device))
    print('Loading model ...\n')
    net = generator(input_dim=args.input_dim, output_dim=args.output_dim,
                    input_size=args.input_size, class_num=args.class_num)
    model = net 
    checkpoint = torch.load(args.pth_path, map_location='cpu')
    checkpoint = proc_nodes_module(checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()
    z,y=prep_preocess(args)
    result = model(z,y)
    result = result.cpu().data.numpy().transpose(0, 2, 3, 1)
    result = (result + 1)/2
    sample_num = args.class_num**2
    image_frame_dim = int(np.floor(np.sqrt(sample_num)))
    if not os.path.exists(os.path.join(args.save_path)):
        os.makedirs(os.path.join(args.save_path))
    utils.save_images(result[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
        os.path.join(args.save_path,'demo_result.png'))
    print("demo image stored in:", os.path.join(args.save_path,'demo_result.png'))
    

if __name__ == "__main__":
    main()