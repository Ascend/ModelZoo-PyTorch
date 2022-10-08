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


import os, sys
import argparse
from math import log10

import torch
from PIL import Image
import numpy as np

import pytorch_ssim
import torchvision.utils as utils
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage, CenterCrop

parser = argparse.ArgumentParser(description='SRGAN get_info script')
parser.add_argument('--data_path', default='./test/data', type=str)
parser.add_argument('--target_path', default='./test/target', type=str)
parser.add_argument('--result_path', default='./dumpOutput_device0', type=str)
# parser.add_argument('--result_path', default='./foward_process', type=str)
parser.add_argument('--Set5_only', default=True, type=bool)
args = parser.parse_args()

# 通过后缀检查是否为图片
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def img_index(file_list, dataset_name, number):
    for i,file in enumerate(file_list):
        if dataset_name in file and number in file:
            return i
    print('No matching pictures in the folder!')
    sys.exit()

# 用于输出图像的转换
def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),  # 把图像调整到400标准格式
        CenterCrop(400),
        ToTensor()
    ])

if __name__ == '__main__':
    print(args)
    if not os.path.exists(args.data_path):
        print('Input data path not exists, please check!')
        sys.exit()
    if not os.path.exists(args.target_path):
        print('Input target path not exists, please check!')
        sys.exit()
    if not os.path.exists(args.result_path):
        print('Input result path not exists, please check!')
        sys.exit()

    lr_path = args.data_path
    hr_path = args.target_path
    bin_path = args.result_path
    output_path = './infer_om_res'

    # 分别读取原始图片和高分辨率图片
    if args.Set5_only:
        lr_file_names = [os.path.join(lr_path, x) for x in os.listdir(lr_path) if
                        is_image_file(x) and x.split('_')[0] == 'Set5']
        hr_file_names = [os.path.join(hr_path, x) for x in os.listdir(hr_path) if
                        is_image_file(x) and x.split('_')[0] == 'Set5']
    else:
        lr_file_names = [os.path.join(lr_path, x) for x in os.listdir(lr_path) if is_image_file(x)]
        hr_file_names = [os.path.join(hr_path, x) for x in os.listdir(hr_path) if is_image_file(x)]

    if len(lr_file_names) < 1 or len(hr_file_names) < 1:
        print('Input data or target have no image data!')
        sys.exit()
    elif len(lr_file_names) != len(hr_file_names):
        print('The number of input data and target do not match!')
        sys.exit()

    # 读取推理结果
    bin_file_names = []
    bin_reduce = []
    fileDir = "result"
    fileList = os.listdir(fileDir)
    for file in fileList:
        bin_dir = os.path.join(fileDir, file)
        bin_file_names.append([os.path.join(bin_dir, x) for x in os.listdir(bin_dir) if x.endswith('.bin')])
    for i in range(len(bin_file_names)):
        bin_reduce.append(bin_file_names[i])
        
    bin_file_names = []
    for _ in bin_reduce:
        bin_file_names += _
        
    tmp = []
    
    for i,_ in enumerate(bin_file_names):
        if _.find("fake_file") < 0:
            tmp.append(_)

    bin_file_names = tmp
    
    if len(bin_file_names) < 1:
        print('Input result path have no result!')
        sys.exit()
        

    psnr_l = []
    ssim_l = []
    for bin_file in bin_file_names:
        bin_name = bin_file.split(os.sep)[-1]
        img_name = bin_name.split('.')[0]
        data_set, img_number = img_name.split('_')[:2]
        print(data_set, img_number)
        lr_img = Image.open(lr_file_names[img_index(lr_file_names,data_set,img_number)]).convert('RGB')
        w,h = lr_img.size
        hr_scale = Resize((2 * h, 2 * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_img)

        hr_img = Image.open(hr_file_names[img_index(hr_file_names,data_set,img_number)]).convert('RGB')
        hr_restore_img = ToTensor()(hr_restore_img).unsqueeze(0)
        hr_img = ToTensor()(hr_img).unsqueeze(0)
        lr_img = ToTensor()(lr_img).unsqueeze(0)

        sr_img = np.fromfile(bin_file, dtype='float32')[: 1*3*2*w*2*h]
        
        sr_img = torch.tensor(sr_img)
        sr_img = sr_img.reshape(hr_img.shape)

        mse = ((hr_img - sr_img) ** 2).data.mean()
        psnr = 10 * log10(1/mse)
        ssim = pytorch_ssim.ssim(sr_img, hr_img).item()

        image_name = '{}_{}.png'.format(data_set, img_number)
        test_images = torch.stack(
            [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_img.data.cpu().squeeze(0)),
             display_transform()(sr_img.data.cpu().squeeze(0))])
        image = utils.make_grid(test_images, nrow=3, padding=5)
        utils.save_image(image, output_path + '/' + image_name.split('.')[0] +
                         '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
                         image_name.split('.')[-1], padding=5)
        psnr_l.append(psnr)
        ssim_l.append(ssim)
        print(f'图片 {image_name} 的生成结果--> PSNR: {psnr:.4f}  SSIM: {ssim:.4f}')
    psnr = round(np.array(psnr_l).mean(),4)
    ssim = round(np.array(ssim_l).mean(),4)
    print(f'Final result in Set--> PSNR: {psnr:.4f}  SSIM: {ssim:.4f}')
