# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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

import argparse
import os
from math import log10

import numpy as np
import torch
if torch.__version__>= "1.8.1":
    print("import torch_npu")
    import torch_npu
import torchvision.utils as utils
from torch.utils.data import DataLoader

import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
import config

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=2, type=int,
                    help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_best.pth', type=str,
                    help='generator model epoch name')
parser.add_argument('--use_npu', default=False, type=bool,
                    help='If use npu for training.')
parser.add_argument('--use_gpu', default=False, type=bool,
                    help='If use gpu for training.')
parser.add_argument('--use_test_code', default=False, type=bool,
                    help='If use test model code')
parser.add_argument('--only_set5', default=True, type=bool,
                    help='If use test model code')
parser.add_argument('--output_dir', default=config.get_root_path(), type=str,
                    help='Path to save running results.')
opt = parser.parse_args()
print('-'*20,'arguments','-'*20)
print(opt)
print('-'*50)

UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name

results = {'Set5': {'psnr': [], 'ssim': []}, 'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
           'Urban100': {'psnr': [], 'ssim': []}, 'SunHays80': {'psnr': [], 'ssim': []}}
# 创建模型
if opt.use_test_code:
    from model_test import Generator
else:
    from model import Generator
model = Generator(UPSCALE_FACTOR).eval()
# 选择模型以及训练设备
if opt.use_npu:
    import torch.npu
    if torch.npu.is_available():
        device = torch.device('npu')
elif opt.use_gpu:
    if torch.cuda.is_available():
        device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'use {device} to run benchmark.')

model.to(device)
# 获取模型的路径
root_dir = opt.output_dir
if not root_dir.endswith('/'):
    root_dir = root_dir + '/'
model_path = root_dir + 'epochs/' + MODEL_NAME
print(f'Load model from: {model_path}')
model.load_state_dict(torch.load(root_dir + 'epochs/' + MODEL_NAME))

test_set = TestDatasetFromFolder('./data/test', upscale_factor=UPSCALE_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=2, batch_size=1, shuffle=False)
# test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

out_path = root_dir + 'benchmark_results_img/'
if not os.path.exists(out_path):
    os.makedirs(out_path)
with torch.no_grad():
    for image_name, lr_image, hr_restore_img, hr_image in test_loader:
        image_name = image_name[0]
        # 判断是否跳过本次循环
        if opt.only_set5 and image_name.split('_')[0] != 'Set5':
            continue
        # lr_image = Variable(lr_image, volatile=True)
        # hr_image = Variable(hr_image, volatile=True)
        lr_image = lr_image.to(device)
        hr_image = hr_image.to(device)

        sr_image = model(lr_image)
        mse = ((hr_image - sr_image) ** 2).data.mean()
        psnr = 10 * log10(1 / mse)
        ssim = pytorch_ssim.ssim(sr_image, hr_image).item()   # 将损失的tensor类型转换为值类型

        test_images = torch.stack(
            [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0)),
             display_transform()(sr_image.data.cpu().squeeze(0))])
        image = utils.make_grid(test_images, nrow=3, padding=5)
        utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
                         image_name.split('.')[-1], padding=5)

        # save psnr\ssim
        results[image_name.split('_')[0]]['psnr'].append(psnr)
        results[image_name.split('_')[0]]['ssim'].append(ssim)
        print(f'img: {image_name}, psnr: {psnr}, ssim: {ssim}')

saved_results = {'psnr': [], 'ssim': []}
for item in results.values():
    psnr = np.array(item['psnr'])
    ssim = np.array(item['ssim'])
    if (len(psnr) == 0) or (len(ssim) == 0):
        psnr = 'No data'
        ssim = 'No data'
    else:
        psnr = round(psnr.mean(), 4)
        ssim = round(ssim.mean(), 4)
    saved_results['psnr'].append(psnr)
    saved_results['ssim'].append(ssim)
print(saved_results)
with open(root_dir + opt.model_name.split('.')[0] + '_test_results.txt', 'w', encoding='utf-8') as f:
    title = 'dataset \t\t psnr \t ssim \n'
    f.write(title)
    if opt.only_set5:
        str ='{0} \t\t  {1} \t {2} \n'.format('Set5',saved_results['psnr'][0], saved_results['ssim'][0])
        f.write(str)
    else:
        for i in range(len(results.keys())):
            if len(list(results.keys())[i]) > 6:
                str = '{0} \t {1} \t {2} \n'.format(list(results.keys())[i],
                                                    saved_results['psnr'][i], saved_results['ssim'][i])
            else:
                str = '{0} \t\t {1} \t {2} \n'.format(list(results.keys())[i],
                                                    saved_results['psnr'][i], saved_results['ssim'][i])
            f.write(str)
