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
import torch
import torch.nn as nn
import torchvision
import torchvision.utils
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.autograd import Variable
import torch.npu
import os
import argparse
import apex
try:
    from apex import amp
except ImportError:
    amp = None

parser = argparse.ArgumentParser(description='VAE+GAN Training')
parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
parser.add_argument('--apex-opt-level', default='O2', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )

NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')
device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')
#去掉异常检测，规避
#torch.autograd.set_detect_anomaly(True)
from dataloader import dataloader
from models import VAE_GAN,Discriminator
from utils import show_and_save,plot_loss

data_loader=dataloader(64)
gen=VAE_GAN().to(f'npu:{NPU_CALCULATE_DEVICE}')
discrim=Discriminator().to(f'npu:{NPU_CALCULATE_DEVICE}')
real_batch = next(iter(data_loader))
show_and_save("training" ,torchvision.utils.make_grid((real_batch[0]*0.5+0.5).cpu(),8))

epochs=25
lr=3e-4
alpha=0.1
gamma=15

criterion=nn.BCELoss().to(f'npu:{NPU_CALCULATE_DEVICE}')
optim_E=apex.optimizers.NpuFusedRMSprop(gen.encoder.parameters(), lr=lr)
optim_D=apex.optimizers.NpuFusedRMSprop(gen.decoder.parameters(), lr=lr)
optim_Dis=apex.optimizers.NpuFusedRMSprop(discrim.parameters(), lr=lr*alpha)
args = parser.parse_args()
if args.apex:
  [gen.encoder,gen.decoder,discrim], [optim_E,optim_D,optim_Dis] = amp.initialize([gen.encoder,gen.decoder,discrim], [optim_E,optim_D,optim_Dis],opt_level=args.apex_opt_level, loss_scale=128, combine_grad=True)
z_fixed=Variable(torch.randn((64,128))).to(f'npu:{NPU_CALCULATE_DEVICE}')
x_fixed=Variable(real_batch[0]).to(f'npu:{NPU_CALCULATE_DEVICE}')

for epoch in range(epochs):
  prior_loss_list,gan_loss_list,recon_loss_list=[],[],[]
  dis_real_list,dis_fake_list,dis_prior_list=[],[],[]
  for i, (data,_) in enumerate(data_loader, 0):
    start_time = time.time()
    bs=data.size()[0]
    ones_label=Variable(torch.ones(bs,1)).to(f'npu:{NPU_CALCULATE_DEVICE}')
    zeros_label=Variable(torch.zeros(bs,1)).to(f'npu:{NPU_CALCULATE_DEVICE}')
    zeros_label1=Variable(torch.zeros(64,1)).to(f'npu:{NPU_CALCULATE_DEVICE}')
    datav = Variable(data).to(f'npu:{NPU_CALCULATE_DEVICE}')
    mean, logvar, rec_enc = gen(datav)
    z_p = Variable(torch.randn(64,128)).to(f'npu:{NPU_CALCULATE_DEVICE}')
    x_p_tilda = gen.decoder(z_p)
    output = discrim(datav)[0]
    errD_real = criterion(output, ones_label)
    dis_real_list.append(errD_real.item())
    output = discrim(rec_enc)[0]
    errD_rec_enc = criterion(output, zeros_label)
    dis_fake_list.append(errD_rec_enc.item())
    output = discrim(x_p_tilda)[0]
    errD_rec_noise = criterion(output, zeros_label1)
    dis_prior_list.append(errD_rec_noise.item())
    gan_loss = errD_real + errD_rec_enc + errD_rec_noise
    gan_loss_list.append(gan_loss.item())
    optim_Dis.zero_grad()
    if args.apex:
      with amp.scale_loss(gan_loss, optim_Dis) as scaled_loss:
        scaled_loss.backward(retain_graph=True)
    else:
      gan_loss.backward(retain_graph=True)
    optim_Dis.step()

    output = discrim(datav)[0]
    errD_real = criterion(output, ones_label)
    output = discrim(rec_enc)[0]
    errD_rec_enc = criterion(output, zeros_label)
    output = discrim(x_p_tilda)[0]
    errD_rec_noise = criterion(output, zeros_label1)
    gan_loss = errD_real + errD_rec_enc + errD_rec_noise
    

    x_l_tilda = discrim(rec_enc)[1]
    x_l = discrim(datav)[1]
    rec_loss = ((x_l_tilda - x_l) ** 2).mean()
    err_dec = gamma * rec_loss - gan_loss 
    recon_loss_list.append(rec_loss.item())
    optim_D.zero_grad()
    if args.apex:
      with amp.scale_loss(err_dec, optim_D) as scaled_loss:
        scaled_loss.backward(retain_graph=True)
    else:
      err_dec.backward(retain_graph=True)
    optim_D.step()
    
    mean, logvar, rec_enc = gen(datav)
    x_l_tilda = discrim(rec_enc)[1]
    x_l = discrim(datav)[1]
    rec_loss = ((x_l_tilda - x_l) ** 2).mean()
    prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
    prior_loss = (-0.5 * torch.sum(prior_loss))/torch.numel(mean.data)
    prior_loss_list.append(prior_loss.item())
    err_enc = prior_loss + 5*rec_loss

    if i > 100:
        pass
   
    optim_E.zero_grad()
    if args.apex:
      with amp.scale_loss(err_enc, optim_E) as scaled_loss:
        scaled_loss.backward(retain_graph=True)
    else:
      err_enc.backward(retain_graph=True)
    optim_E.step()
    FPS = bs / (time.time() - start_time)
    if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_gan: %.4f\tLoss_prior: %.4f\tRec_loss: %.4f\tdis_real_loss: %0.4f\tdis_fake_loss: %.4f\tdis_prior_loss: %.4f\tFPS: %.4f'
                  % (epoch,epochs, i, len(data_loader),
                     gan_loss.item(), prior_loss.item(),rec_loss.item(),errD_real.item(),errD_rec_enc.item(),errD_rec_noise.item(),FPS))

  

  b=gen(x_fixed)[2]
  b=b.detach()
  c=gen.decoder(z_fixed)
  c=c.detach()
  show_and_save('MNISTrec_noise_epoch_%d.png' % epoch ,torchvision.utils.make_grid((c*0.5+0.5).cpu(),8))
  show_and_save('MNISTrec_epoch_%d.png' % epoch ,torchvision.utils.make_grid((b*0.5+0.5).cpu(),8))

plot_loss(prior_loss_list)
plot_loss(recon_loss_list)
plot_loss(gan_loss_list)

