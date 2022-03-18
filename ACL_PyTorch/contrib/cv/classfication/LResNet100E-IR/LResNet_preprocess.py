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
# limitations under the License

import os
import sys
sys.path.append("./LResNet")
import pickle
from glob import glob

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as trans


def jpg2bin(data_path, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    bins, issame_list = pickle.load(open(data_path, 'rb'), encoding='bytes')
    for i in range(len(bins)):
        _bin = bins[i]
        img_np_arr = np.frombuffer(_bin, np.uint8)
        img = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
        img = Image.fromarray(img.astype(np.uint8))
        img = transform(img)
        img_flip = torch.flip(img, dims=[2])
        img = img.numpy()
        img_flip = img_flip.numpy()

        img.tofile(os.path.join(save_dir, str(i) + ".bin"))
        img_flip.tofile(os.path.join(save_dir, str(i) + "_flip" + ".bin"))

        if (i+1) % 1000 == 0:
            print('loading bin', (i+1))

    np.save('data/lfw_list', np.array(issame_list))
    print('success load data')


def bin2info(bin_dir, data_info, width, height):
    bin_images = glob(os.path.join(bin_dir, '*.bin'))
    with open(data_info, 'w') as file:
        for index, img in enumerate(bin_images):
            content = ' '.join([str(index), img, width, height])
            file.write(content)
            file.write('\n')


if __name__ == '__main__':
    file_type = sys.argv[1]
    data_path = sys.argv[2]
    info_path = sys.argv[3]
    if file_type == 'bin':
        width = sys.argv[4]
        height = sys.argv[5]
        assert len(sys.argv) == 6, 'The number of input parameters must be equal to 5'
        bin2info(data_path, info_path, width, height)
    elif file_type == 'jpg':
        assert len(sys.argv) == 4, 'The number of input parameters must be equal to 3'
        jpg2bin(data_path, info_path)

