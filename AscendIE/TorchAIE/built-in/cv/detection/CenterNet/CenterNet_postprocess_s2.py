# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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
import argparse
from glob import glob

import torch
import numpy as np
import cv2
from tqdm import tqdm

ROOT = './CenterNet/src/'
if ROOT not in sys.path:
    sys.path.append(ROOT)

from lib.opts import opts
from lib.detectors.detector_factory import detector_factory
from lib.datasets.dataset_factory import get_dataset
from lib.models.decode import ctdet_decode
from lib.utils.post_process import ctdet_post_process
from lib.models.model import create_model, load_model
import lib.datasets.dataset.coco
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CenterNet')
    parser.add_argument('--dataset', default='/data/datasets', type=str, help='dataset')
    parser.add_argument('--resultfolder', default='./run_eval_result', type=str, help='Dir to save results')
    parser.add_argument('--postprocessed_dir', default='./postprocessed', type=str, help='Dir that contains postprocessed results')
    args = parser.parse_args()

    new_datapath = args.dataset
    if not os.path.exists(args.resultfolder):
        os.makedirs(args.resultfolder)
    opt = opts().parse('{} --load_model {}'.format('ctdet', './ctdet_coco_dla_2x.pth').split(' '))
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    opt.data_dir = new_datapath
    dataset = Dataset(opt, 'val')
    opt.gpus[0] = -1 
    results = {}
    num_iters = len(dataset)

    for ind in tqdm(range(num_iters)):
       img_id = dataset.images[ind]
       ret = np.load(os.path.join(args.postprocessed_dir,str(img_id)+'.npz'),allow_pickle=True)['dic'].tolist()
       results[img_id] = ret
    dataset.run_eval(results, args.resultfolder)