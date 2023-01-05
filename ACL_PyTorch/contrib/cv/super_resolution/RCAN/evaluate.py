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


import os
import sys
import math
import json
import argparse

import cv2
from PIL import Image
import numpy as np

from skimage.metrics import structural_similarity


def rgb2y_matlab(x):
    """Convert RGB image to illumination Y in Ycbcr space in matlab way.
    -------------
    # Args
        - Input: x, byte RGB image, value range [0, 255]
        - Ouput: byte gray image, value range [16, 235] 

    # Shape
        - Input: (H, W, C)
        - Output: (H, W) 
    """
    K = np.array([65.481, 128.553, 24.966]) / 255.0
    Y = 16 + np.matmul(x, K)
    return Y


def PSNR(im1, im2, use_y_channel=True):
    """Calculate PSNR score between im1 and im2
    --------------
    # Args
        - im1, im2: input byte RGB image, value range [0, 255]
        - use_y_channel: if convert im1 and im2 to illumination channel first
    """
    if use_y_channel:
        im1 = rgb2y_matlab(im1)
        im2 = rgb2y_matlab(im2)
    im1 = im1.astype(np.float)[3:-1-3, 3:-1-3]
    im2 = im2.astype(np.float)[3:-1-3, 3:-1-3]
    mse = np.mean(np.square(im1 - im2)) 
    return 10 * np.log10(255 ** 2 / mse) 


def SSIM(gt_img, noise_img):
    """Calculate SSIM score between im1 and im2 in Y space
    -------------
    # Args
        - gt_img: ground truth image, byte RGB image
        - noise_img: image with noise, byte RGB image
    """
    gt_img = rgb2y_matlab(gt_img).astype(np.uint8)
    noise_img = rgb2y_matlab(noise_img).astype(np.uint8)
     
    ssim_score = structural_similarity(gt_img, noise_img, gaussian_weights=True, 
            sigma=1.5, use_sample_covariance=False)
    return ssim_score


def evaluate(infer_img_path, HR_img_path):
    infer_list = os.listdir(infer_img_path)
    HR_img_list = os.listdir(HR_img_path)
    val_PSNR = 0
    val_SSIM = 0

    for file in infer_list:
        if not file in HR_img_list:
            print('{} not found in HR_img_path'.format(file))
            sys.exit()
        infer_img = Image.open(os.path.join(infer_img_path, file))
        HR_img = Image.open(os.path.join(HR_img_path, file))
        val_PSNR += PSNR(infer_img, HR_img)
        val_SSIM += SSIM(infer_img, HR_img)
    
    val_PSNR /= len(infer_list)
    val_SSIM /= len(infer_list)
    print('Evaluation of RCAN model')
    print('PSNR\t{}'.format(val_PSNR))
    print('SSIM\t{}'.format(val_SSIM))
