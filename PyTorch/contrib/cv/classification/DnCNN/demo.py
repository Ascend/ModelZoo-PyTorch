# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
# -*- coding: utf-8 -*- 

import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from urllib.request import urlretrieve
from skimage.measure import compare_psnr

def batch_PSNR(img, imclean, data_range):
    """ comprare two data """
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])


class DnCNN(nn.Module):
    """ DnCnn class """
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, \
                    kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, \
                    kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, \
                    kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        
    def forward(self, x):
        """ forward train """
        out = self.dncnn(x)
        return out


def main():
    """ check one pic """
    global deviceType
    
    #deviceType='npu:0'  
    deviceType='cpu'
    net = DnCNN(channels=1, num_of_layers=17)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids)
    #model = nn.DataParallel(net, device_ids=device_ids).npu()
    torch.device(deviceType)
    model.load_state_dict(torch.load("net.pth")) #加载模型的位置
    
    print("model get")
    
    #下载一张在线图片
    with open('url.ini', 'r') as f:
        content = f.read()
        img_url = content.split('img_url=')[1].split('\n')[0]
    IMAGE_URL =  img_url
    urlretrieve(IMAGE_URL,"tem.png")
    
    model.eval()
    
    img = cv2.imread("tem.png")
    im_h = img.shape[0]
    im_w = img.shape[1]
    
    imgA = np.float32(img[:, :, 0])
    imgB = np.float32(img[:, :, 1])
    imgC = np.float32(img[:, :, 2])
    
    imgA -= modelOneChannle(imgA/255, model)
    imgB -= modelOneChannle(imgB/255, model)
    imgC -= modelOneChannle(imgC/255, model)
    
    cl_im = np.zeros((im_h, im_w, 3))
    for tm_h in range(im_h):
        for tm_w in range(im_w):
            cl_im[tm_h][tm_w][0] = imgA[tm_h][tm_w]  #blue
            cl_im[tm_h][tm_w][1] = imgB[tm_h][tm_w]   #Green
            cl_im[tm_h][tm_w][2] = imgC[tm_h][tm_w]        #red
    
    
    cv2.imwrite("clear.png", cl_im)
    

def modelOneChannle(imgTmp, model):
    """ model one channel change """
    imgTmp = np.expand_dims(imgTmp, 0)
    imgTmp = np.expand_dims(imgTmp, 1)
    
    imgTmp = torch.Tensor(imgTmp) 
    imgTmp = imgTmp.to(deviceType)
    imgTmp = model(imgTmp)
    
    imgTmp = imgTmp.cpu().detach().numpy()

    imgTmp = np.squeeze(imgTmp)
    
    return imgTmp
    
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
