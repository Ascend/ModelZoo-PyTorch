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
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import logging

from  torch.nn import Sequential, ConvTranspose2d, UpsamplingBilinear2d, \
    BatchNorm2d, ReLU, Tanh, Linear
from copy import copy 

import utils

logging.basicConfig(filename="./log_generator.txt", filemode='w',level=logging.INFO)
class Generator(torch.nn.Module):
    def __init__(self,config):
        super(Generator,self).__init__()

        self.parse_config(config)
        self.generator = Sequential()
        
        #first layer. Project and reshape noise
        self.c_latent = self.g_feature_size * (2**(self.g_layers-1))
        self.latent_hw = int(self.img_h/(2**self.g_layers))
        self.generator.add_module('TConvLatent',ConvTranspose2d(self.c_input,self.c_latent,self.latent_hw,bias=False))
        self.generator.add_module('BNLatent',BatchNorm2d(self.c_latent))
        self.generator.add_module('ReLULatent',ReLU(inplace=True))
        
        c_input = self.c_latent

        layer_number = 1
        for i in range(self.g_layers-2,-1,-1):
            c_layer = int(self.g_feature_size *(2**i))
            self.generator.add_module('TConv'+str(layer_number),ConvTranspose2d(c_input,c_layer,self.kernel_size, self.stride,self.g_input_pad,output_padding=self.g_output_pad,bias=False))
            self.generator.add_module('BN'+str(layer_number),BatchNorm2d(c_layer))
            self.generator.add_module('ReLU'+str(layer_number),ReLU(inplace=True))
            c_input = copy(c_layer)
            layer_number+=1

        #final image layer
        self.generator.add_module('F_TConv1',ConvTranspose2d(c_input,self.img_c,self.kernel_size, self.stride, self.g_input_pad, output_padding=self.g_output_pad,bias=False))
        #no batch norm in output layer acc to DCGAN paper
        self.generator.add_module('F_Tanh',Tanh())

    
    def parse_config(self, config):
        self.g_feature_size=config['g_feature_size']
        self.g_layers = config['g_layers']
        self.len_z=config['len_z']
        self.img_h=config['img_h']
        self.img_w=config['img_w']
        self.img_c=config['img_c']
        self.c_input = config['len_z']
        self.stride = config['g_stride']
        self.kernel_size = config['g_kernel_size']
        self.g_input_pad = config['g_input_pad']
        self.g_output_pad = config['g_output_pad']
    
    def forward(self,z):

        generated_image = self.generator(z)

        return generated_image