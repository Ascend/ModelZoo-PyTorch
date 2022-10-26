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
import os.path
import numpy as np
import random
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import data_augmentation
from torch.autograd import Variable
import sys

def normalize(data):
    return data/255.

def preprocess(data_path = './dataset/BSD68',save_path='./prep_dataset'):
    files_source = glob.glob(os.path.join(data_path, '*.png'))
    files_source.sort()
    # process data
    psnr_test = 0 
    for f in files_source:
        # image
        Img = cv2.imread(f)
        H = Img.shape[0]
        W = Img.shape[1]
        if H > W:
            Img= cv2.flip(cv2.transpose(Img), 1)
        Img = normalize(np.float32(Img[:,:,0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        # noise
        torch.manual_seed(0) #set the seed
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=25/255.)
        # noisy image
        INoisy = ISource + noise
        ISource = Variable(ISource) 
        INoisy = Variable(INoisy) 
        print(f,'has benn transformed into binary file')
        name = (f.split('/')[-1]).split('.')[0]
        ISource = np.array(ISource).astype(np.float32)
        if not os.path.exists(os.path.join(save_path,'ISoure')):
            os.makedirs(os.path.join(save_path,'ISoure'))
        if not os.path.exists(os.path.join(save_path,'INoisy')):
            os.makedirs(os.path.join(save_path,'INoisy'))
        ISource.tofile(os.path.join(save_path,'ISoure',name+'.bin'))
        INoisy = np.array(INoisy).astype(np.float32)
        INoisy.tofile(os.path.join(save_path,'INoisy',name+'.bin'))
if __name__ == '__main__':
    data_path = sys.argv[1]
    save_path = sys.argv[2]
    preprocess(data_path,save_path)
