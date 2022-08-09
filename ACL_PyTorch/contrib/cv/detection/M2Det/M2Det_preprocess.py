'''
# Copyright 2020 Huawei Technologies Co., Ltd
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
'''

import os
import sys
sys.path.insert(0, './M2Det')
import warnings
warnings.filterwarnings('ignore')
import torch
import argparse
import numpy as np
from layers.functions import Detect, PriorBox
from data import BaseTransform
from configs.CC import Config
from utils.core import get_dataloader, print_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='M2Det Preprocess')
    parser.add_argument('-c', '--config', default='../configs/m2det512_vgg.py', type=str)
    parser.add_argument('-d', '--dataset', default='COCO', help='VOC or COCO version')
    parser.add_argument('--test', action='store_true', help='to submit a test file')
    parser.add_argument("--save_folder", default="./pre_dataset")
    parser.add_argument('--COCO_imgs', default="~/data/coco/images", help='COCO images root')
    parser.add_argument('--COCO_anns', default="~/data/coco/annotations", help='COCO annotations root')
    args = parser.parse_args()
    
    cfg = Config.fromfile(args.config)
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    
    _set = 'eval_sets' if not args.test else 'test_sets'
    testset = get_dataloader(args, cfg, args.dataset, _set)

    _preprocess = BaseTransform(cfg.model.input_size, cfg.model.rgb_means, (2, 0, 1))
   # _preprocess = BaseTransform(cfg.model.input_size, cfg.model.rgb_means, (0,1,2))
    num_images = len(testset)
    print_info('=> Total {} images to test.'.format(num_images), ['yellow', 'bold'])

    for i in range(num_images):
        input_image, img_id= testset.pull_image(i)
        img_name = img_id.split('/')[-1]
        print(img_name, "===", i)
        input_tensor = _preprocess(input_image).unsqueeze(0)
        img = np.array(input_tensor).astype(np.float32)
    #    img = np.array(input_tensor).astype(np.uint8)
        img.tofile(os.path.join(args.save_folder, img_name.split('.')[0] + ".bin"))
        
