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

from __future__ import print_function
import os
import argparse
import torch
import numpy as np
import cv2


parser = argparse.ArgumentParser(description='FaceBoxes')

parser.add_argument('--dataset', default='data/FDDB')
parser.add_argument('--save-folder', default='prep/')
    
args = parser.parse_args()


if __name__ == '__main__':
    
        
    # testing dataset
    testset_folder = os.path.join(args.dataset, 'images/')
    testset_list = os.path.join('data/FDDB', 'img_list.txt')
    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)
    
    save_info_path = os.path.join(args.save_folder)
    if not os.path.exists(save_info_path):
        os.makedirs(save_info_path)   
    fw = open(os.path.join(args.save_folder, 'FDDB.txt'), 'w')
    
    
    # testing begin
    for i, img_name in enumerate(test_dataset):
        if i<3000:   
            image_path = testset_folder + img_name + '.jpg'
            img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = np.float32(img_raw)
            
            target_size = 1024
            im_shape = img.shape
            print("assert:", im_shape[0:2])
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            
            resize = target_size / im_size_max
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)
            width_pad = target_size - img.shape[1]
            left = 0 
            right = width_pad
            height_pad = target_size -img.shape[0]
            top = 0
            bottom = height_pad
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
            im_height, im_width, _ = img.shape
    
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            
            img_name1='_'.join(img_name.split('/')) 
            print('begin: {0}, {1}'.format(img_name,img.shape))
            fw.write('{:s} {:.3f} {:.3f} {:.3f}\n'.format(img_name1,im_height, im_width, resize))
            
            
            true_path = os.path.join(args.save_folder)
            if not os.path.exists(true_path):
                os.makedirs(true_path)
            img.numpy().tofile(os.path.join(true_path, img_name1+'.bin'))
            
        
    fw.close()
    print('Preprocessing completed!')