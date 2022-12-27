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
import sys
import glob
import argparse
from math import log10

import torch
import numpy as np
import torchvision.utils as utils
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage, CenterCrop
from PIL import Image
from tqdm import tqdm

import pytorch_ssim


parser = argparse.ArgumentParser(description='SRGAN get_acc_info script')
parser.add_argument('--hr_path', default='./datasets/Set5', type=str)
parser.add_argument('--result_path', default='./result/bs1', type=str)
parser.add_argument('--save_path', default='./infer_om_res', type=str)
args = parser.parse_args()


if __name__ == '__main__':
    if not os.path.exists(args.hr_path):
        print('Input hr_path not exists, please check!')
        exit(-1)
    if not os.path.exists(args.result_path):
        print('Input result path not exists, please check!')
        exit(-1)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    hr_path = args.hr_path
    bin_path = args.result_path
    output_path = args.save_path

    psnr_sum = 0
    ssim_sum = 0

    hr_file_names = os.listdir(hr_path)
    bin_file_names = glob.glob(f'{bin_path}/*/*.bin')
    
    name_map = dict(map(lambda x:(x.split(".")[0], x) , hr_file_names))
    
    for bin_file in bin_file_names:
        bin_name = bin_file.split(os.sep)[-1]
        img_name = bin_name[:-6]
        print(bin_file.split(os.sep)[-1][:-6])
        image_name = name_map[img_name]
        
        try:
            hr_img = Image.open(os.path.join(hr_path, image_name)).convert('RGB')
        except:
            print(f"-------- ERROR!please check that the input picture does not match the input infer result! --------")
            exit(-1)

        hr_img = ToTensor()(hr_img).unsqueeze(0)
        hr_img_size = list(hr_img.shape)

        sr_img = np.fromfile(bin_file, dtype='float32')
        sr_img_tensor = torch.tensor(sr_img)
        sr_img_tensor = sr_img_tensor.reshape(hr_img_size)
        
        mse = ((hr_img - sr_img_tensor) ** 2).data.mean()
        psnr = 10 * log10(1/mse)
        ssim = pytorch_ssim.ssim(sr_img_tensor, hr_img).item()

        psnr_sum += psnr
        ssim_sum += ssim
        print(f'the result of the {image_name} picture --> PSNR: {psnr:.4f}  SSIM: {ssim:.4f}')

        image = ToPILImage(mode="RGB")(torch.squeeze(sr_img_tensor))
        image.save(os.path.join(output_path, image_name))

    length = len(bin_file_names)
    psnr = round(psnr_sum / length, 4)
    ssim = round(ssim_sum / length, 4)
    print(f'---------------------------------------------------------')
    print(f'Final result in Set--> PSNR: {psnr:.4f}  SSIM: {ssim:.4f}')
