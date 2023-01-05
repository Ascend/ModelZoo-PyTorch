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

# -*- coding:GB2312 -*- 

from __future__ import absolute_import
from __future__ import division

import sys
import os
import cv2
import torch

from lib.opts_pose import opts
from lib.detectors.detector_factory import detector_factory
from lib.datasets.dataset_factory import get_dataset
    
def preprocess(file_path, bin_path):
    opt = opts().parse('--task {} --load_model {}'.format('multi_pose', 'model_best.pth').split(' ')) 
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    in_files = os.listdir(file_path)
    
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)
    for file in sorted(in_files):
       os.chdir(os.path.join(file_path, file))
       cur_path = os.getcwd()
       doc = os.listdir(cur_path)   
       for document in doc:
          if document=='output':
              break
          image = cv2.imread(os.path.join(cur_path, document))
          for scale in opt.test_scales:
             images, meta = detector.pre_process(image, scale, meta=None)
             images.numpy().tofile(os.path.join(bin_path,document.split('.')[0] +'.bin'))
        
if __name__ == "__main__":
    file_path = os.path.abspath(sys.argv[1])
    bin_path = os.path.abspath(sys.argv[2])
    preprocess(file_path, bin_path)
