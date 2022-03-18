"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
#
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
#
"""

import torch
from models.networks.base_network import BaseNetwork
from models.networks.loss import *
from models.networks.discriminator import *
from models.networks.generator import *
from models.networks.encoder import *
import util.util as util


def find_network_using_name(target_network_name, filename):
    target_class_name = target_network_name + filename
    module_name = 'models.networks.' + filename
    network = util.find_class_in_module(target_class_name, module_name)

    assert issubclass(network, BaseNetwork), \
        "Class %s should be a subclass of BaseNetwork" % network

    return network


def modify_commandline_options(parser, is_train):
    opt, _ = parser.parse_known_args()

    netG_cls = find_network_using_name(opt.netG, 'generator')
    parser = netG_cls.modify_commandline_options(parser, is_train)
    if is_train:
        netD_cls = find_network_using_name(opt.netD, 'discriminator')
        parser = netD_cls.modify_commandline_options(parser, is_train)
    netE_cls = find_network_using_name('conv', 'encoder')
    parser = netE_cls.modify_commandline_options(parser, is_train)

    return parser


def create_network(cls, opt):
    net = cls(opt)
    net.print_network()
    if len(opt.gpu_ids) > 0:
        assert(torch.npu.is_available())
        net.npu()
    net.init_weights(opt.init_type, opt.init_variance)
    return net


def define_G(opt):
    netG_cls = find_network_using_name(opt.netG, 'generator')
    return create_network(netG_cls, opt)


def define_D(opt):
    netD_cls = find_network_using_name(opt.netD, 'discriminator')
    return create_network(netD_cls, opt)


def define_E(opt):
    # there exists only one encoder type
    netE_cls = find_network_using_name('conv', 'encoder')
    return create_network(netE_cls, opt)
