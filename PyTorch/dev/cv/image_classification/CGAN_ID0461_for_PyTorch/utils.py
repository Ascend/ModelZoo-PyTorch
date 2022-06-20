# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
import yaml
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch.nn as nn
import matplotlib.animation as animation
import torchvision.utils as vutils
from matplotlib.animation import PillowWriter

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def load_config(file_name):
    config=yaml.safe_load(open(file_name))
    return config

def set_manual_seed(config):
    if 'reproduce' in config and config['reproduce']==True:
        seed = 0 if 'seed' not in config else config['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

def load_transformed_dataset(config):
    h = config['img_h']
    w = config['img_w']
    mean = config['input_normalise_mean']
    std = config['input_normalise_std']

    dataset = ImageFolder(config['data_path'], 
                        transform=transforms.Compose([
                            transforms.Resize(h),#to maintain aspect ratio
                            transforms.CenterCrop((h,w)),
                            transforms.ToTensor(),#normalize takes a tensor
                            transforms.Normalize((mean,mean,mean),(std,std,std))
                            ]))
    return dataset

def get_dataloader(config, dataset):
    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                        shuffle=config['shuffle'], num_workers=config['num_workers'])
    return dataloader

def init_weights(m):
    '''
    if type(m) in [ torch.nn.ConvTranspose2d, torch.nn.Conv2d, torch.nn.Linear ]:
        torch.nn.init.normal_(m.weight,0.0,0.02)
    if type(m) == torch.nn.BatchNorm2d: #copied this from tutorial. Need to figure out the logic
        torch.nn.init.normal_(m.weight,1,0.02)
        torch.nn.init.constant_(m.bias,0)
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_loss_plot(g_loss,d_loss):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_loss,label="G")
    plt.plot(d_loss,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./loss.png")

def save_result_images(real_images,fake_images,nrow,config):
    mean = config['input_normalise_mean']
    std = config['input_normalise_std']
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    grid_real = vutils.make_grid(real_images*std+mean,nrow=nrow)
    plt.axis("off")
    plt.title("Real Images")
	################################ modify by npu ##########################################
    plt.imshow(grid_real.permute(1,2,0))
	################################ modify by npu ##########################################

    plt.subplot(1,2,2)
    grid_fake = vutils.make_grid(fake_images*std+mean,nrow=nrow)
    plt.axis("off")
    plt.title("Fake Images")
	################################ modify by npu ##########################################
    plt.imshow(grid_fake.permute(1,2,0))
    plt.savefig("./generated_images.png",dpi=300)
	################################ modify by npu ##########################################

def save_gif(image_array,nrow,config):
    mean = config['input_normalise_mean']
    std = config['input_normalise_std']
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    imgs = [vutils.make_grid(i.cpu()*std+mean,nrow=nrow) for i in image_array]
    ims = [[plt.imshow(np.transpose(i.cpu(),(1,2,0)), animated=True)] for i in imgs]
    ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=1000, blit=True)

	################################ modify by npu ##########################################
    ani.save('./animation.gif', writer=PillowWriter(fps=10))
	################################ modify by npu ##########################################
