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
import numpy as np
import os
import sys
sys.path.append('./RefineDet.PyTorch')
from data import VOCAnnotationTransform, VOCDetection, BaseTransform
from data import voc_refinedet

dataset_mean = (104, 117, 123)
cfg = voc_refinedet['320']



if __name__ == '__main__':

    datasets_path, save_folder = sys.argv[1:3]
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    dataset = VOCDetection(root=datasets_path,
                           image_sets=[('2007', 'test')],
                           transform=BaseTransform(320, dataset_mean),
                           target_transform=VOCAnnotationTransform(),
                           dataset_name='VOC07test')

    for i in range(len(dataset)):
        im, gt, h, w = dataset.pull_item(i)

        name = '%07d'%(i + 1)

        img_name = name + '.bin'

        img_save_path = os.path.join(save_folder, img_name)
        im = np.array(im).astype(np.float32)
        im.tofile(img_save_path)

        print(i)
    print('all finish!')

