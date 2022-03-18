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
import sys
import os
import os.path
import numpy as np
import random
import torch
import cv2
import glob

infer_data = 'Set68'
infer_noiseL = 15

def normalize(data):
    return data / 255.


def proprecess(data_path, ISource_bin, INoisy_bin):

    # load data info
    print('Loading data info ...\n')
    files = glob.glob(os.path.join(data_path, infer_data, '*.png'))
    files.sort()
    # process data
    for i in range(len(files)):
        # image
        filename = os.path.basename(files[i])
        img = cv2.imread(files[i])
        img = normalize(np.float32(img[:, :, 0]))

        img_padded = np.full([481, 481], 0, dtype=np.float32)
        width_offset = (481 - img.shape[1]) // 2
        height_offset = (481 - img.shape[0]) // 2
        img_padded[height_offset:height_offset + img.shape[0], width_offset:width_offset + img.shape[1]] = img
        img = img_padded

        img = np.expand_dims(img, 0)
        img = np.expand_dims(img, 1)

        ISource = torch.Tensor(img)
        # noise
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=infer_noiseL / 255.)
        # noisy image
        INoisy = ISource + noise

        # save ISource_bin
        ISource = ISource.numpy()
        print("ISource shape is", ISource.shape)
        ISource.tofile(os.path.join(ISource_bin, filename.split('.')[0] + '.bin'))
        
        # save INoisy_bin
        INoisy = INoisy.numpy()
        print("INoisy shape is", INoisy.shape) 
        INoisy.tofile(os.path.join(INoisy_bin, filename.split('.')[0] + '.bin'))
        
if __name__ == '__main__':
    
    data_path = sys.argv[1]
    ISource_bin =  sys.argv[2]
    INoisy_bin = sys.argv[3]
    if os.path.exists(ISource_bin) is False:
        os.mkdir(ISource_bin)
    if os.path.exists(INoisy_bin) is False:
        os.mkdir(INoisy_bin)

    proprecess(data_path, ISource_bin, INoisy_bin)
