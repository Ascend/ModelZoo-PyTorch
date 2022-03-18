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
from CGAN import generator as G
import os
import numpy as np
import torch
import struct
import glob
import sys
import utils
import argparse


def parse_args():
    desc = "Pytorch implementation of CGAN collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--bin_out_path', type=str, default='', help="the output inferenced")
    parser.add_argument('--save_path', type=str, default='result', help="the generated image path")
    return parser.parse_args()


def get_save_path(bin_folder):
    result_paths = []
    files_source = glob.glob(os.path.join(bin_folder,'*.bin'))
    files_source.sort()
    for file in files_source:
        if file.endswith('.bin'):
            result_path = file
            result_paths.append(result_path)
    return result_paths
    
        
def file2tensor(output_bin):
    size = os.path.getsize(output_bin)
    res1 = []
    L = int(size / 4)
    binfile = open(output_bin, 'rb')
    for i in range(L):
        data = binfile.read(4)
        num = struct.unpack('f', data)
        res1.append(num[0])
    binfile.close()
    dim_res = np.array(res1).reshape(100,3,28,28)
    tensor_res = torch.tensor(dim_res, dtype=torch.float32)
    return tensor_res


def post_process(args):
    result_paths = get_save_path(args.bin_out_path)
    for i in range(len(result_paths)):
        result = file2tensor(result_paths[i])
        result = result.data.numpy().transpose(0, 2, 3, 1)
        result = (result + 1)/2
        sample_num = 100
        image_frame_dim = int(np.floor(np.sqrt(sample_num)))
        if not os.path.exists(os.path.join(args.save_path)):
            os.makedirs(os.path.join(args.save_path))
        utils.save_images(result[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          os.path.join(args.save_path,'result.png'))
        print("postprocess image stored in:", os.path.join(args.save_path,'result.png'))

if __name__ == "__main__":
    args = parse_args()
    post_process(args)
