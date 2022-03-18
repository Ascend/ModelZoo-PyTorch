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
from collections import OrderedDict
import torch
from models import networks_adapt as networks


class load_networks():
    def __init__(self, opt):
        self.opt = opt
        self.gpu = 0
        self.netG_A = networks.define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.netG,
                                        self.opt.norm, not self.opt.no_dropout, self.opt.init_type, self.opt.init_gain,
                                        self.gpu)
        self.netG_B = networks.define_G(self.opt.output_nc, self.opt.input_nc, self.opt.ngf, self.opt.netG,
                                        self.opt.norm, not self.opt.no_dropout, self.opt.init_type, self.opt.init_gain,
                                        self.gpu)
        if (opt.npu == False):
            self.device = torch.device('cuda:{}'.format(self.gpu))
        else:
            self.device = torch.device("cpu")

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def proc_nodes_module(self, checkpoint):
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if "module." in k:
                name = k.replace("module.", "")
            else:
                name = k
            new_state_dict[name] = v
        return new_state_dict

    def loadnetworks(self, net, load_path):
        state_dict = torch.load(load_path, map_location=torch.device('cpu'))
        state_dict = self.proc_nodes_module(state_dict)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        net.load_state_dict(state_dict)
        return net

    def get_networks(self, load_patha, load_pathb):
        model_Ga = self.loadnetworks(self.netG_A, load_patha)
        model_Gb = self.loadnetworks(self.netG_B, load_pathb)
        return model_Ga, model_Gb
