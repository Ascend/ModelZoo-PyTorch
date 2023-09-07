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
import argparse
from glob import glob

import torch
import numpy as np
import cv2
from tqdm import tqdm

from lib.opts import opts
from lib.detectors.detector_factory import detector_factory
from lib.datasets.dataset_factory import get_dataset
from lib.models.decode import ctdet_decode
from lib.utils.post_process import ctdet_post_process
from lib.models.model import create_model, load_model
import lib.datasets.dataset.coco


def post_process(dets, meta, scale=1):
    num_classes=80
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], 80)
    for j in range(1, 81):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
      dets[0][j][:, :4] /= scale
    return dets[0]  
        

def merge_outputs(detections):
    results = {}
    for j in range(1, 80 + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
    return results


def run(result_list, index, meta, filenames):
    output={}
    for i in range(3):
        buf = np.fromfile(f'{result_list}/{filenames[0:-4]}_{i}.bin', dtype="float32")
        if i == 0:
           output['hm'] = torch.tensor(buf.reshape(1, 80, 128, 128))
        if i == 1:
           output['wh'] = torch.tensor(buf.reshape(1, 2, 128, 128))
        if i == 2:
           output['reg'] = torch.tensor(buf.reshape(1, 2, 128, 128))
    detections = []
    hm = output['hm'].sigmoid_()
    wh = output['wh']
    reg = output['reg']
    detss = ctdet_decode(hm, wh, reg) 
    dets = post_process(detss, meta)
    detections.append(dets)
    results = merge_outputs(detections)
    return results
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CenterNet')
    parser.add_argument('--bin_data_path', default='./result/dumpOutput_device0', type=str, help='infer out path')
    parser.add_argument('--dataset', default='/opt/npu', type=str, help='dataset')
    args = parser.parse_args()
    new_datapath = args.dataset
    opt = opts().parse('{} --load_model {}'.format('ctdet', './ctdet_coco_dla_2x.pth').split(' '))
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    opt.data_dir = new_datapath
    Detector = detector_factory[opt.task]
    dataset = Dataset(opt, 'val')
    opt.gpus[0] = -1 
    detector = Detector(opt)
    filename = []
    num_iters = len(dataset)
    for ind in tqdm(range(num_iters)):
       img_id = dataset.images[ind]
       img_info = dataset.coco.loadImgs(ids=[img_id])[0]
       img_path = os.path.join(dataset.img_dir, img_info['file_name'])
       image = cv2.imread(img_path)
       images, metas = detector.pre_process(image, 1.0, meta=None)
       ret = run(args.bin_data_path, ind, metas, img_info['file_name'])
       np.savez(os.path.join('save', str(img_id)), dic=ret)