# Copyright 2022 Huawei Technologies Co., Ltd
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

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import cv2
import numpy as np
import os
import sys
import json
import argparse
from tqdm import tqdm
import time
np.set_printoptions(threshold=np.inf)

from mmocr.models.builder import POSTPROCESSOR
from mmocr.models.textdet.postprocess.base_postprocessor import BasePostprocessor
from mmocr.models.textdet.postprocess.utils import fill_hole, fourier2poly, poly_nms

from mmocr.utils import check_argument
from mmocr.models.textdet.postprocess import FCEPostprocessor
        
def resize_boundary(boundaries, scale_factor):
    assert check_argument.is_2dlist(boundaries)
    assert isinstance(scale_factor, np.ndarray)
    assert scale_factor.shape[0] == 4

    for b in boundaries:
        sz = len(b)
        check_argument.valid_boundary(b, True)
        b[:sz -1] = (np.array(b[:sz - 1]) *
                (np.tile(scale_factor[:2], int(
                    (sz - 1) / 2)).reshape(1, sz - 1))).flatten().tolist()
    return boundaries

    
def get_boundary(scales, score_maps, scale_factor, rescale):
    assert len(score_maps) == len(scales)
    boundaries = []
    for idx, score_map in enumerate(score_maps):
        scale = scales[idx]
        boundaries = boundaries + get_boundary_single(score_map, scale)

        # nms
    boundaries = poly_nms(boundaries, 0.1)

    if rescale:
        boundaries = resize_boundary(boundaries, 1.0 / scale_factor)

    results = dict(boundary_result=boundaries)
    return results

def get_boundary_single(score_map, scale):
    assert len(score_map) == 2
    postprocessor = FCEPostprocessor(fourier_degree = 5,
                     text_repr_type='poly',
                     num_reconstr_points=50,
                     alpha=1.0,
                     beta=2.0,
                     score_thr=0.3)
    return postprocessor(score_map, scale)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',type=str,default='./result/')
    parser.add_argument('--instance_file',type=str,default='./mmocr/data/icdar2015/instances_test.json')
    parser.add_argument('--output_file',type=str,default='./boundary_results.txt')
    args = parser.parse_args()
    #print(args.input_path)
    #print(args.output_file)
    #prediction_file_path = './result/'
    prediction_file_path = args.input_path
    for root, dirs, files in os.walk(prediction_file_path):
        prediction_file_path = os.path.join(prediction_file_path,dirs[0])
        break
    container = []
    count = 0
    
    for i in range(501):
         container.append([])
         for j in range(3):
             container[i].append([])
             for k in range(2):
                 container[i][j].append([])
                 
    img_idx = []
    #file_name = './mmocr/data/icdar2015/instances_test.json'
    file_name = args.instance_file
    img_name = []

    file_info = json.load(open(file_name, 'r'))
    filename = file_info["images"]
    length = len(filename)
    for i in range(0,length):
        img_name.append(filename[i]['file_name'])

    for enum in img_name:
        index1 = enum.rfind('_')
        index2 = enum.rfind('.')
        img_num = enum[index1+1:index2]
        img_idx.append(int(img_num))
    
    for tfile_name in os.listdir(prediction_file_path):
        tmp = tfile_name.split('.')[0]
        index = tmp.rfind('_')
        img_name = tmp[:index]
        index1 = img_name.rfind('_')
        img_name = tmp[:index1]

        index2 = img_name.rfind('_')+1
        flag = int(img_name[index2:])
        
        lines = ''
        with open(os.path.join(prediction_file_path,tfile_name), 'r') as f:
            for line in f.readlines():
                line = line.strip()
                lines = lines+' '+line
            temp = lines.strip().split(" ")
            l = len(temp)
            temp = np.array(temp)
            temp = list(map(float,temp))
            temp = torch.Tensor(temp)
            cont0 = []
            if l == 11360:
                temp = temp.reshape(1,4,40,71)
                container[flag][2][0] = temp
                count += 1
            elif l == 45440:
                temp = temp.reshape(1,4,80,142)
                container[flag][1][0] = temp
                count += 1
            elif l == 181760:
                temp = temp.reshape(1,4,160,284)
                container[flag][0][0] = temp
                count += 1
            elif l == 62480:
                temp = temp.reshape(1,22,40,71)
                container[flag][2][1] = temp
                count += 1
            elif l == 249920:
                temp = temp.reshape(1,22,80,142)
                container[flag][1][1] = temp
                count += 1
            elif l == 999680:
                temp = temp.reshape(1,22,160,284)
                container[flag][0][1] = temp
                count += 1     
            print("\r", end="")
            i = int(count/30)
            print("process: {}%: ".format(i), ">" * (i // 2), end="")
            sys.stdout.flush()
            time.sleep(0.05)
    postprocess = FCEPostprocessor(fourier_degree = 5,
                     text_repr_type='poly',
                     num_reconstr_points=50,
                     alpha=1.0,
                     beta=2.0,
                     score_thr=0.3)
    scale = 5
    scales = (8, 16, 32)
    rescale = True
    scale_factor = np.array([1.765625 , 1.7652777, 1.765625 , 1.7652777])
    for i in range(0,500):
        idx = img_idx[i]
        score_maps = container[idx]    
        result = get_boundary(scales, score_maps, scale_factor, rescale)
        save_path = args.output_file
        f=open(save_path,"a+")
        f.writelines(str(result)+'\n')