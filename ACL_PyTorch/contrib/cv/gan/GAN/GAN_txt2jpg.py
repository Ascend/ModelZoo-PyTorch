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

import argparse
import torch
import numpy as np
import os
from torchvision.utils import save_image

def read_bin(filename):

    data = np.fromfile(filename,dtype=np.float32)
    data = torch.Tensor(data)
    data = data.view(-1,1,28,28)
    return data

def main(args):
    old_path = os.listdir(args.txt_path)
    os.makedirs(args.infer_results_path, exist_ok=True)
    old_path.sort(reverse=True)
    new_path = args.txt_path+'/'+old_path[0]
    files = os.listdir(new_path)
    for file in files:
        filename = new_path + '/' + file
        data = read_bin(filename)
        if file[1]!='_':
            save_path = args.infer_results_path + '/' + file[:2] + ".jpg"
        else:
            save_path = args.infer_results_path + '/' + file[0] + ".jpg"

        save_image(data, save_path,normalize=True)
    print("done!")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_path', type=str, required=True)
    parser.add_argument('--infer_results_path', type=str, required=True)
    args = parser.parse_args()
    main(args)
