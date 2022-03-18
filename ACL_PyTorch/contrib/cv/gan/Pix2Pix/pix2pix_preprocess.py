# Copyright 2021 Huawei Technologies Co., Ltd
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

from PIL import Image
import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from options.test_options import TestOptions
import random
import torch
import numpy as np
# python pix2pix_preprocess.py --dataroot ./datasets/facades/ 

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_torch()

def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print ("---  new folder...  ---")
 
	else:
		print ("---  There is this folder!  ---")
		


def preprocess(opt,AB):
    w, h = AB.size
    w2 = int(w / 2)
    A = AB.crop((0, 0, w2, h))
    B = AB.crop((w2, 0, w, h))
    transform_params = get_params(opt, A.size)
    # A_transform = get_transform(opt, transform_params, grayscale=(3 == 1))
    B_transform = get_transform(opt, transform_params, grayscale=(3 == 1))
    # A = A_transform(A)
    B = B_transform(B)
    return  B# 0.9686 # A[1,35,46] tensor(-0.1451)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    src_path = os.path.join(opt.dataroot,'test')
    save_path = os.path.join(opt.dataroot,'bin')
    # src_path = './datasets/facades/test'
    # save_path = './datasets/facades/bin'
    
    mkdir(save_path) 

    in_files = os.listdir(src_path)
  
  
    for idx, file in enumerate(in_files):
        idx = idx + 1
        print(file, "===", idx)
        input_image = Image.open(src_path + '/' + file).convert('RGB')
        input_tensor = preprocess(opt,input_image)
        img = np.array(input_tensor).astype(np.float32)
        img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))
