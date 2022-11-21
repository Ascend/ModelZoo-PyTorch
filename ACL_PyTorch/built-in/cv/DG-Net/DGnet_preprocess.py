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

from __future__ import print_function
import sys
sys.path.append('./DG-Net')
from trainer import DGNet_Trainer, to_gray
from utils import get_config
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import random
import os
import numpy as np
from torchvision import datasets, models, transforms
from PIL import Image
from shutil import copyfile
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--output_folder', type=str, default="./bin_path1", help="output image path")
parser.add_argument('--output_folder2', type=str, default="./bin_path2", help="output image path")
parser.add_argument('--input_folder', type=str, default="../Market/pytorch/train_all/", help="input image path")

parser.add_argument('--name', type=str, default="E0.5new_reid0.5_w30000", help="model name")

parser.add_argument('--batchsize', default=1, type=int, help='batchsize')

if __name__ == '__main__':
    opts = parser.parse_args()
    opts.config = './outputs/%s/config.yaml'%opts.name
    config = get_config(opts.config)

    if not os.path.exists(opts.output_folder):
        os.makedirs(opts.output_folder)
    else:
        os.system('rm -rf %s/*'%opts.output_folder)

    if not os.path.exists(opts.output_folder2):
        os.makedirs(opts.output_folder2)
    else:
        os.system('rm -rf %s/*'%opts.output_folder2)

    data_transforms = transforms.Compose([
            transforms.Resize(( config['crop_image_height'], config['crop_image_width']), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
        img_flip = img.index_select(3,inv_idx)
        return img_flip

    image_datasets = datasets.ImageFolder(opts.input_folder, data_transforms)
    dataloader_content = torch.utils.data.DataLoader(image_datasets, batch_size=opts.batchsize, shuffle=False, pin_memory=True, num_workers=1)
    dataloader_structure = torch.utils.data.DataLoader(image_datasets, batch_size=opts.batchsize, shuffle=True, pin_memory=True, num_workers=1)
    image_paths = image_datasets.imgs

    gray = to_gray(False)
    pbar = tqdm.tqdm(total=len(dataloader_content))
    for data, data2, path in zip(dataloader_content, dataloader_structure, image_paths):
        name = os.path.basename(path[0])
        id_img, label = data
        id_img_flip = Variable(fliplr(id_img).cpu())
        id_img = Variable(id_img.cpu())
        bg_img, label2 = data2
        bg_img = gray(bg_img)
        bg_img = Variable(bg_img.cpu())
        dst_path = opts.output_folder
        dst_path2 = opts.output_folder2
        id_img.numpy().tofile(dst_path + '/%03d_%03d_gan%s.bin'%(label2, label, name[:-4])) 
        bg_img.numpy().tofile(dst_path2 + '/%03d_%03d_gan%s.bin'%(label2, label, name[:-4])) 
        pbar.update(1)

    pbar.close()