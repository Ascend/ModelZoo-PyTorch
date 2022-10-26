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

def postProcessing(folder_path, batch_size):
    resultPath = "./output_bs{}/jpg".format(batch_size)
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)

    for cnt in range(0, 64):
        x_fake_list = []
        for i in range(0, 5):
            x_new = read_txt(os.path.join(folder_path, str(cnt*5 + i) + "_0.txt"), batch_size)
            x_fake_list.append(x_new)
            x_concat = torch.cat(x_fake_list, dim=3)
        result_path = os.path.join(resultPath, '{}-images.jpg'.format(cnt))
        save_image(denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
        print('Saved real and fake images into {}...'.format(result_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    postProcessing(args.folder_path, args.batch_size)