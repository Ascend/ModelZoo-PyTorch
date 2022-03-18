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
import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN
from utils import AverageMeter, weights_init_kaiming, batch_PSNR
from torch.nn.parallel import DistributedDataParallel as DDP
import apex
from apex import amp 
import argparse

parser = argparse.ArgumentParser(description='DnCNN')
parser.add_argument('--data_path', type=str, help='path of dataset')
parser.add_argument('--resume', type=str, help='path of pre_trained model')
opt = parser.parse_args()

def main():
    """ check one pic """
    #模型定义和设备初始化
    deviceType='npu:0'
    #deviceType='cpu'
    net = DnCNN(channels=1, num_of_layers=17)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).npu()
    torch.device(deviceType)
    model.load_state_dict(torch.load(opt.resume)) #加载模型的位置
    
    print("model get")
    
    #读出图片
    allFiles = glob.glob(os.path.join(opt.data_path, "Set68/*.png"))
    allFiles.sort()
    allPsnr=0
    allStep=0
    model.eval()
    for file in allFiles:
        img = cv2.imread(file)
        img = np.float32(img[:, :, 0])/255
        
        #img_padded = np.full([481, 481], 0, dtype=np.float32)
        #width_offset = (481 - img.shape[1]) // 2
        #height_offset = (481 - img.shape[0]) // 2
        #img_padded[height_offset:height_offset + img.shape[0], width_offset:width_offset + img.shape[1]] = img
        #img = img_padded
                
        img = np.expand_dims(img, 0)
        img = np.expand_dims(img, 1)
        ISource = torch.Tensor(img)
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=15 / 255.)
        ISource, noise = ISource.to(deviceType), noise.to(deviceType)
        INoisy = ISource + noise
        model = model.to(deviceType)
        print("model evl")
        with torch.no_grad(): # this can save much memory
            Out = torch.clamp(INoisy - model(INoisy), 0., 1.)
        psnr = batch_PSNR(Out, ISource, 1.)
        print("file: {}  psnr :{:.3f}".format(file, psnr))
        allPsnr += psnr
        allStep += 1
    avg = allPsnr / allStep
    print("eval avg psnr = {:.3f}".format(avg))
    
if __name__ == "__main__":
    main()



