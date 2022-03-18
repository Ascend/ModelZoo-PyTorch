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
import argparse
import torch
import os


class parse_args():
    def __init__(self, isTrain=True, isTest=False):
        self.isTrain = isTrain
        self.isTest = isTest
        self.parser = argparse.ArgumentParser(description='Pytorch CycleGAN training')

    def initialize(self):
        parser = self.parser
        parser.add_argument('--npu', default=False, help='whether to use npu to fastern training')
        parser.add_argument('--pu_ids', type=str, default='1',
                            help='gpu ids(npu ids): e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--dataroot', type=str, default='./datasets/maps',
                            help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='maps_cycle_gan',
                            help='name of the experiment. It decides where to store samples and models')

        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='cycle_gan',
                            help='chooses which model to use. [cycle_gan]')
        parser.add_argument('--input_nc', type=int, default=3,
                            help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3,
                            help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic',
                            help='specify discriminator architecture [basic | n_layers | pixel]. '
                                 'The basic model is a 70x70 PatchGAN. n_layers allows you to'
                                 ' specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_9blocks',
                            help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance',
                            help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--batch_size', type=int, default=1,
                            help='batch_size')
        # additional parameters
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--model_ga_path', type=str,
                            default='./checkpoints/maps_cycle_gan/latest_net_G_A.pth',
                            help='path for modelga')
        parser.add_argument('--model_gb_path', type=str,
                            default='./checkpoints/maps_cycle_gan/latest_net_G_B.pth',
                            help='path for modelga')
        parser.add_argument('--onnx_path', type=str,
                            default='./onnxmodel/',
                            help='path for modelga')
        parser.add_argument('--model_ga_onnx_name', type=str,
                            default='model_Ga.onnx',
                            help='onnx name for modelga')
        parser.add_argument('--model_gb_onnx_name', type=str,
                            default='model_Gb.onnx',
                            help='onnx for modelgb')
        parser.add_argument('--gpuPerformance', type=str,
                            default='./gpuPerformance/',
                            help='file for t4 test result ')
        parser.add_argument('--npu_bin_file', type=str,
                            default='./result/dumpOutput_device0/',
                            help='npu bin ')
        parser.add_argument('--bin2img_fie', type=str,
                            default='./bin2imgfile/',
                            help='save bin2img  ')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        parser = parser.parse_args()
        parser.process_device_map = self.device_id_to_process_device_map(parser.pu_ids)
        return parser

    def device_id_to_process_device_map(self, device_list):
        devices = device_list.split(",")
        devices = [int(x) for x in devices]
        devices.sort()

        process_device_map = dict()
        for process_id, device_id in enumerate(devices):
            process_device_map[process_id] = device_id
        return process_device_map

    def change_parser(self, isTrain=True, isTest=False):
        self.isTest = isTest
        self.isTrain = isTrain
        self.parser = None
        return self.initialize()

    def printParser(self):
        pasers = self.parser.parse_args()
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(pasers).items()):
            comment = ''
            default = self.parser.get_default(k)
            # if v != default:
            #     comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
