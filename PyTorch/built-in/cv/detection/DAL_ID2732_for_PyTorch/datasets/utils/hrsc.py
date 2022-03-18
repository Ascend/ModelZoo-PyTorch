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
import cv2
import torch
import shutil
from shutil import copyfile
import numpy as np
from tqdm import tqdm 


def mkdirs(dst_dir):
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs('{}/{}'.format(dst_dir, 'Train/AllImages'))
    os.makedirs('{}/{}'.format(dst_dir, 'Train/Annotations'))
    os.makedirs('{}/{}'.format(dst_dir, 'Test/AllImages'))
    os.makedirs('{}/{}'.format(dst_dir, 'Test/Annotations'))


def copy_data(root_dir, dst_dir):
    with open(os.path.join(root_dir, 'ImageSets/trainval.txt'), 'r') as f:
        lines = f.readlines()
        trainval = [x[:-1] for x in lines]
    test = [y[:-4] for y in os.listdir(os.path.join(root_dir, 'Test/AllImages'))]
    assert len(trainval)==617 and len(test)==444, 'errors occurred in data copy process'
    print('creat train dataset...')
    for train_id in tqdm(trainval):
        src_img = os.path.join(root_dir, 'Train/AllImages/'+train_id+'.bmp')
        src_xml = os.path.join(root_dir, 'Train/Annotations/'+train_id+'.xml')
        dst_img = os.path.join( dst_dir, 'Train/AllImages/'+train_id+'.bmp')
        dst_xml = os.path.join( dst_dir, 'Train/Annotations/'+train_id+'.xml')
        copyfile(src_img, dst_img)
        copyfile(src_xml, dst_xml)
    print('creat test dataset...')
    for test_id in tqdm(test):
        src_img = os.path.join(root_dir, 'Test/AllImages/'+test_id+'.bmp')
        src_xml = os.path.join(root_dir, 'Test/Annotations/'+test_id+'.xml')
        dst_img = os.path.join( dst_dir, 'Test/AllImages/'+test_id+'.bmp')
        dst_xml = os.path.join( dst_dir, 'Test/Annotations/'+test_id+'.xml')
        copyfile(src_img, dst_img)
        copyfile(src_xml, dst_xml)
    instance_statistics(dst_dir+'Train/Annotations/', dst_dir+'Test/Annotations/')

def instance_statistics(*paths):
    for path in paths:
        cnt = 0
        labels = os.listdir(path)
        for label in labels:
            with open(os.path.join(path, label), 'r') as f:
                contents = f.read()
                objects = contents.split('<HRSC_Object>')
                cnt += len(objects) - 1
        print('dataset:{} \nimages:{} \ntargets:{}\n\n'.format(path, len(labels), cnt))



def creat_dataset(root_dir, dst_dir):
    assert os.path.exists(root_dir), 'Root dir not exist, check you path !'
    mkdirs(dst_dir)
    copy_data(root_dir, dst_dir)
    

if __name__ == "__main__":
    root_dir = '/py/DATASET/HRSC2016'
    dst_dir = '/py/BoxesCascade/HRSC2016'
    creat_dataset(root_dir, dst_dir)    # 

