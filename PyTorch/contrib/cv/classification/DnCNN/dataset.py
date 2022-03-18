# -*- coding: utf-8 -*- 
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import os.path
import numpy as np
import random
from numpy.core.fromnumeric import clip
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import changePose, data_augmentation
import time
 
def normalize(data):
    """ change pic info into float """
    return data / 255.


def Im2Patch(img, win, stride=1):
    """ sample image """
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 1:stride, 0:endh - win + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)
    
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


class Traindata(udata.Dataset):
    """ dataset for trian """
    def __init__(self, data_path, getDataSet='train'):
        # train
        self.scales = [1, 0.9, 0.8, 0.7]
        self.files = glob.glob(os.path.join(data_path, getDataSet, '*.png'))
        self.files.sort()

        self.cDataArray = []
        self.noiseArray = []
        self.nDatarray = []

        for fileName in self.files:
            img = cv2.imread(fileName)
            print(fileName)
            for sca in self.scales:
                h, w, c = img.shape
                npImg = cv2.resize(img, (int(h * sca), int(w * sca)), interpolation=cv2.INTER_CUBIC)
                npImg = np.expand_dims(npImg[:, :, 0].copy(), 0)
                npImg = np.float32(normalize(npImg))
                patches = Im2Patch(npImg, win=5, stride=10)

                for n in range(patches.shape[3]):
                    one = patches[:, :, :, n].copy()
                    one = data_augmentation(one, np.random.randint(1, 8))
                    clear_data = torch.Tensor(one.copy())
                    noise = torch.FloatTensor(clear_data.size()).normal_(mean=0, std= 15. / 255.)
                    nosie_dada = clear_data + noise 
                    self.cDataArray.append(clear_data)
                    self.nDatarray.append(nosie_dada)
                    self.noiseArray.append(noise)
        
        print('create data')

    def __getitem__(self, index):
        return self.nDatarray[index], self.cDataArray[index], self.noiseArray[index]
       
    def __len__(self):
        return len(self.cDataArray)


class Evldata(udata.Dataset):
    """ dataset for reasioning """
    def __init__(self, data_path, getDataSet):
        self.myData=[]
        # train
        print('eval data')
        self.files = glob.glob(os.path.join(data_path, getDataSet, '*.png'))
        self.files.sort()

    def __getitem__(self, index):
        img = self.files[index]
        img = cv2.imread(img)
        h, w, c = img.shape
        npImg = cv2.resize(img, (int(h), int(w)))
        npImg = np.expand_dims(npImg[:, :, 0].copy(), 0)
        npImg = np.float32(normalize(npImg))

        clear_data = torch.Tensor(npImg)
        noise = torch.FloatTensor(clear_data.size()).normal_(mean=0, std= 15. / 255.)
        nosie_dada = clear_data + noise 
                
        return nosie_dada, clear_data, noise

    def __len__(self):
        return len(self.files)
    


  
        
        
        
        
        
        
        
