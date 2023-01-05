# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -*- coding: UTF-8 -*- 
#!/usr/bin/env python
import re
from PIL import Image
import numpy as np
import os
import argparse
import shutil

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_path', type=str,default='./datasets_test')
    parser.add_argument('--source_path', type=str,default='/home/test_user06/temp_store/CelebA_img')
    parser.add_argument("--target_path", type=str,default='./test_imgs_')
    opts = parser.parse_args()
    
    
    path_img1= opts.source_path
    path_img2= opts.target_path
    test_path = opts.test_path
    os.makedirs(path_img2, exist_ok=True)
    
    file_name_list = os.listdir(test_path)
    for file_name in file_name_list:
        path1 = os.path.join(test_path,file_name)
        data = []
        with open(path1,'r') as fr:
            data = fr.readlines()
            data = ''.join(data).strip('\n').splitlines() 
        

        for name in data:
            name1=name.split(" ")[0]
            (filepath, filename) = os.path.split(name1)
            path_old=os.path.join(path_img1,filename)
            path_new=os.path.join(path_img2,filename)
            shutil.copy(path_old,path_new)
