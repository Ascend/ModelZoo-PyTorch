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

from  torch.nn import Sequential, Conv2d, UpsamplingBilinear2d, \
    BatchNorm2d, LeakyReLU, Tanh, Linear, Sigmoid
from copy import copy 

import utils

logging.basicConfig(filename="./log_discriminator.txt", filemode='w',level=logging.INFO)
class Discriminator(torch.nn.Module):
    def __init__(self,config):
        super(Discriminator,self).__init__()

        self.parse_config(config)
        self.discriminator = Sequential()
        self.final_layer = Sequential()
        
        #first image layer
        c_layer = self.g_feature_size
        self.discriminator.add_module('Conv1',Conv2d(self.img_c, c_layer,self.kernel_size, self.stride, self.g_input_pad,bias=False))
        #no batch norm in input payer acc to paper
        self.discriminator.add_module('LeakyReLU',LeakyReLU(self.leaky_slope,inplace=True))

        layer_number = 2
        for i in range(1,self.g_layers):
            c_input = copy(c_layer)
            c_layer = int(self.g_feature_size *(2**i))
            self.discriminator.add_module('Conv'+str(layer_number),Conv2d(c_input,c_layer,self.kernel_size, self.stride,self.g_input_pad,bias=False))
            self.discriminator.add_module('BN'+str(layer_number),BatchNorm2d(c_layer))
            self.discriminator.add_module('LeakyReLU'+str(layer_number),LeakyReLU(self.leaky_slope,inplace=True))
            layer_number+=1        

        #flatten and sigmoid
        height = int(self.img_h/2**self.g_layers)
        self.final_layer.add_module('MapTo1', Conv2d(c_layer,1,height,bias=False))
        self.final_layer.add_module('Sigmoid', Sigmoid())
    
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
        self.leaky_slope = config['leaky_ReLU_slope']
    
    def forward(self,images):
        logging.info("Input Shape = " + str(images.shape))
        logging.info(self.discriminator)
        feature_cube = self.discriminator(images)

        #decide if data image or generated image
        decision = self.final_layer(feature_cube)
        decision = decision.reshape(decision.shape[0],-1)   #shape[0]=batch size

        return decision