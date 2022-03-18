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
from torch.autograd import Variable
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder,self).__init__()
    self.conv1=nn.Conv2d(1,64,5,padding=2,stride=2)   #in_channels=3
    self.bn1=nn.BatchNorm2d(64,momentum=0.9)
    self.conv2=nn.Conv2d(64,128,5,padding=2,stride=2)
    self.bn2=nn.BatchNorm2d(128,momentum=0.9)
    self.conv3=nn.Conv2d(128,256,5,padding=2,stride=2)
    self.bn3=nn.BatchNorm2d(256,momentum=0.9)
    self.relu=nn.LeakyReLU(0.2)
    self.fc1=nn.Linear(256*8*8,2048)
    self.bn4=nn.BatchNorm1d(2048,momentum=0.9)
    self.fc_mean=nn.Linear(2048,128)
    self.fc_logvar=nn.Linear(2048,128)   #latent dim=128
  
  def forward(self,x):
    batch_size=x.size()[0]
    out=self.relu(self.bn1(self.conv1(x)))
    out=self.relu(self.bn2(self.conv2(out)))
    out=self.relu(self.bn3(self.conv3(out)))
    out=out.view(batch_size,-1)
    out=self.relu(self.bn4(self.fc1(out)))
    mean=self.fc_mean(out)
    logvar=self.fc_logvar(out)
    
    return mean,logvar
class Decoder(nn.Module):
  def __init__(self):
    super(Decoder,self).__init__()
    self.fc1=nn.Linear(128,8*8*256)
    self.bn1=nn.BatchNorm1d(8*8*256,momentum=0.9)
    self.relu=nn.LeakyReLU(0.2)
    self.deconv1=nn.ConvTranspose2d(256,256,6, stride=2, padding=2)
    self.bn2=nn.BatchNorm2d(256,momentum=0.9)
    self.deconv2=nn.ConvTranspose2d(256,128,6, stride=2, padding=2)
    self.bn3=nn.BatchNorm2d(128,momentum=0.9)
    self.deconv3=nn.ConvTranspose2d(128,32,6, stride=2, padding=2)
    self.bn4=nn.BatchNorm2d(32,momentum=0.9)
    self.deconv4=nn.ConvTranspose2d(32,1,5, stride=1, padding=2)
    self.tanh=nn.Tanh()

  def forward(self,x):
    batch_size=x.size()[0]
    x=self.relu(self.bn1(self.fc1(x)))
    x=x.view(-1,256,8,8)
    x=self.relu(self.bn2(self.deconv1(x)))
    x=self.relu(self.bn3(self.deconv2(x)))
    x=self.relu(self.bn4(self.deconv3(x)))
    x=self.tanh(self.deconv4(x))
    return x
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator,self).__init__()
    self.conv1=nn.Conv2d(1,32,5,padding=2,stride=1)
    self.relu=nn.LeakyReLU(0.2)
    self.conv2=nn.Conv2d(32,128,5,padding=2,stride=2)
    self.bn1=nn.BatchNorm2d(128,momentum=0.9)
    self.conv3=nn.Conv2d(128,256,5,padding=2,stride=2)
    self.bn2=nn.BatchNorm2d(256,momentum=0.9)
    self.conv4=nn.Conv2d(256,256,5,padding=2,stride=2)
    self.bn3=nn.BatchNorm2d(256,momentum=0.9)
    self.fc1=nn.Linear(8*8*256,512)
    self.bn4=nn.BatchNorm1d(512,momentum=0.9)
    self.fc2=nn.Linear(512,1)
    self.sigmoid=nn.Sigmoid()

  def forward(self,x):
    batch_size=x.size()[0]
    x=self.relu(self.conv1(x))
    x=self.relu(self.bn1(self.conv2(x)))
    x=self.relu(self.bn2(self.conv3(x)))
    x=self.relu(self.bn3(self.conv4(x)))
    x=x.view(-1,256*8*8)
    x1=x;
    x=self.relu(self.bn4(self.fc1(x)))
    x=self.sigmoid(self.fc2(x))

    return x,x1
class VAE_GAN(nn.Module):
  def __init__(self):
    super(VAE_GAN,self).__init__()
    self.encoder=Encoder()
    self.decoder=Decoder()
    self.discriminator=Discriminator()
    self.encoder.apply(weights_init)
    self.decoder.apply(weights_init)
    self.discriminator.apply(weights_init)


  def forward(self,x):
    bs=x.size()[0]
    z_mean,z_logvar=self.encoder(x)
    std = z_logvar.mul(0.5).exp_()
    #sampling epsilon from normal distribution
    epsilon=Variable(torch.randn(bs,128)).to(f'npu:{NPU_CALCULATE_DEVICE}')
    z=z_mean+std*epsilon
    x_tilda=self.decoder(z)
      
    return z_mean,z_logvar,x_tilda
    
    
