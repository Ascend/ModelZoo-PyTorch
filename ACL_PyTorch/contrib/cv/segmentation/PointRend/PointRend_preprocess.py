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
import sys 
import numpy as np
import torch

from detectron2.structures import ImageList
from detectron2.data import transforms as T
import detectron2.data.detection_utils as utils


def normalize(img):
    pixel_mean = np.array([103.5300, 116.2800, 123.6750], dtype=np.float32).reshape(3, 1, 1)
    pixel_std = np.array([1., 1., 1.], dtype=np.float32).reshape(3, 1, 1)
    return (img - pixel_mean) / pixel_std


def preprocess(src_path, sav_path):
    citys = sorted(os.listdir(src_path))
    augmentation = T.ResizeShortestEdge(1024, 2048, 'choice')
    trans = augmentation.get_transform(torch.zeros((1024, 2048, 3)))
    idx = 0
    for city in citys:
        city_path = os.path.join(src_path, city)
        files = sorted(os.listdir(city_path))
        for file in files:
            print('preprocessing image {}'.format(idx + 1), end='\r')
            image = utils.read_image(os.path.join(city_path, file), format='BGR')
            image = trans.apply_image(image)
            image = normalize(image.transpose(2, 0, 1))
            image = [torch.as_tensor(image)]
            image = ImageList.from_tensors(image, 32).tensor.squeeze().numpy()
            image.tofile(os.path.join(sav_path, file.split('.')[0] + '.bin'))
            idx += 1


if __name__ == '__main__':
    src_path = sys.argv[1] + '/leftImg8bit/val'
    sav_path = sys.argv[2]
    if not os.path.exists(sav_path):
        os.makedirs(sav_path)
    
    preprocess(src_path, sav_path)