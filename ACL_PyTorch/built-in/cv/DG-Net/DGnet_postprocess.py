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
parser.add_argument('--output_folder', type=str, default="./off-gan_id/", help="output image path")
parser.add_argument('--output_folder2', type=str, default="./off-gan_bg/", help="output image path")
parser.add_argument('--input_folder', type=str, default="../Market/pytorch/train_all/", help="input image path")
parser.add_argument('--result_folder', type=str, default="/home/shikang/DG-net/DG-Net/2022_11_14-11_19_44/", help="input image path")
parser.add_argument('--name', type=str, default="E0.5new_reid0.5_w30000", help="model name")
parser.add_argument('--which_epoch', default=100000, type=int, help='iteration')

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

	def recover(inp):
	    """Imshow for Tensor."""
	    inp = inp.transpose((1, 2, 0))
	    mean = np.array([0.485, 0.456, 0.406])
	    std = np.array([0.229, 0.224, 0.225])
	    inp = std * inp + mean
	    inp = inp * 255.0
	    inp = np.clip(inp, 0, 255)
	    return inp

	for im in os.listdir(opts.result_folder):
		if im != 'sumary.json':
		    label = im.split('_')[0]
		    label2 = im.split('_')[1]
		    name = '_'.join(im.split('_')[2:6])
		    dst_path = opts.output_folder + '/%s'%label
		    dst_path2 = opts.output_folder2 + '/%s'%label2
		    if not os.path.isdir(dst_path):
		        os.mkdir(dst_path)
		    if not os.path.isdir(dst_path2):
		        os.mkdir(dst_path2)
		    outputs = np.fromfile(os.path.join(opts.result_folder, im),dtype=np.float32).reshape(1,3,256,128)
		    im = recover(outputs[0])
		    im = Image.fromarray(im.astype('uint8'))  
		    im.save(dst_path + '/%s_%s_gan%s.jpg'%(label2, label, name))
		    im.save(dst_path2 + '/%s_%s_gan%s.jpg'%(label2, label, name))
