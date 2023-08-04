# Copyright 2023 Huawei Technologies Co., Ltd
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
from argparse import ArgumentParser
from utils import DefaultBoxes, Encoder, COCODetection
from utils import SSDTransformer
from ssd_r34 import SSD_R34
import torch
import numpy as np
import tqdm

def parse_args():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")
    parser.add_argument('--data', '-d', type=str, default='../coco',
                        help='path to test and training data files')
    parser.add_argument('--image-size', default=[1200,1200], type=int, nargs='+',
                        help='input image sizes (e.g 1400 1400,1200 1200')  
    parser.add_argument('--strides', default=[3,3,2,2,2,2], type=int, nargs='+',
                        help='stides for ssd model must include 6 numbers')                                                             
    parser.add_argument('--output_dir',  type=str, default=None,
                        help='predata files')
    return parser.parse_args()


def dboxes_R34_coco(figsize, strides):
    ssd_r34=SSD_R34(81, strides=strides)
    synt_img=torch.rand([1,3]+figsize)
    _,_,feat_size =ssd_r34(synt_img, extract_shapes = True)
    steps=[(int(figsize[0]/fs[0]),int(figsize[1]/fs[1])) for fs in feat_size]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [(int(s*figsize[0]/300),int(s*figsize[1]/300)) for s in [21, 45, 99, 153, 207, 261, 315]] 
    aspect_ratios =  [[2], [2, 3], [2, 3], [2, 3], [2], [2]] 
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


def preprocess(coco, output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir, mode=0o640)

    for idx, image_id in enumerate(tqdm.tqdm(coco.img_keys)):

        img, (htot, wtot), _, _ = coco[idx]
       
        inp_data_np = np.expand_dims(np.array(img), axis = 0)

        output_file = f'{output_dir}/{image_id:0>12}_{htot}_{wtot}.npy'

        np.save(output_file, inp_data_np)

def main():
    args = parse_args()

    dboxes = dboxes_R34_coco(args.image_size, args.strides)
    val_trans = SSDTransformer(dboxes, (args.image_size[0], args.image_size[1]), val=True)

    val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.data, "val2017")

    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)

    preprocess(val_coco, args.output_dir)

if __name__ == "__main__":
    main()