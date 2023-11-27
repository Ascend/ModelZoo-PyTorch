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

import torch
import argparse
from torchvision.utils import save_image
import numpy as np
import os

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def read_txt(path, batch_size):
    data = []
    data_bat = []
    with open(path, 'r') as file_to_read:
        for line in file_to_read.readlines():
            data_tmp = [float(i) for i in line.split()]
            data_bat.append(data_tmp)
            if len(data_bat) == 128:
                data.append(data_bat)
                data_bat = []
    data = np.array(data)
    data = torch.Tensor(data)
    data = data.view(batch_size, 3, 128, 128)
    return data

def postProcessing(batch_size, ts_model_path, input_bin_folder_path):
    # resultPath = "./output_bs{}/jpg".format(batch_size)
    # if not os.path.exists(resultPath):
    #     os.makedirs(resultPath)

    # for cnt in range(0, 64):
    #     x_fake_list = []
    #     for i in range(0, 5):
    #         x_new = read_txt(os.path.join(folder_path, str(cnt*5 + i) + "_0.txt"), batch_size)
    #         x_fake_list.append(x_new)
    #         x_concat = torch.cat(x_fake_list, dim=3)
    #     result_path = os.path.join(resultPath, '{}-images.jpg'.format(cnt))
    #     save_image(denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
    #     print('Saved real and fake images into {}...'.format(result_path))
    
    ts_model = torch.jit.load(ts_model_path)
    
    list_files_attr_bin = os.path.join(input_bin_folder_path, "attr")
    list_files_img_bin = os.path.join(input_bin_folder_path, "img")
    for attr_binfile_name in os.listdir(list_files_attr_bin):
        
        attr_bin_filepath = os.path.join(list_files_attr_bin, attr_binfile_name)
        attr_np_arr = np.fromfile(attr_bin_filepath).reshape((1, 5))
        attr_tensor = torch.tensor(attr_np_arr, dtype=torch.float32)
        
        img_bin_filepath = os.path.join(list_files_img_bin, attr_binfile_name)
        img_np_arr = np.fromfile(img_bin_filepath).reshape((1, 3, 128, 128))
        img_tensor = torch.tensor(img_np_arr, dtype=torch.float32)
        
        res = ts_model(img_tensor, attr_tensor)
        print(res.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--folder_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--ts_model_path", default="./stargan.ts")
    parser.add_argument("--input_bin_folder_path", default="./bin")
    args = parser.parse_args()
    postProcessing(args.batch_size, args.ts_model_path, args.input_bin_folder_path)