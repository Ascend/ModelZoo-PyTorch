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


import torch
import utils
import numpy as np
from torch.autograd import Variable

class Prefetcher(object):
    def __init__(self, loader, noises, stream=None):
        self.loader = iter(loader)
        self.noises = noises
        self.stream = stream if stream is not None else torch.npu.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return

        with torch.npu.stream(self.stream):
            next_input_list = []
            noise_list = []
            for int_noise_sigma in self.noises:
                noise_sigma = int_noise_sigma / 255
                new_images = utils.add_batch_noise(self.next_input, noise_sigma)
                noise_sigma = torch.FloatTensor(np.array([noise_sigma for idx in range(new_images.shape[0])]))
                new_images = Variable(new_images).npu(non_blocking=True)
                noise_sigma = Variable(noise_sigma).npu(non_blocking=True)
                next_input_list.append(new_images)
                noise_list.append(noise_sigma)
            self.next_input = self.next_input.npu(non_blocking=True)
            self.new_images = next_input_list
            self.noise_sigma = noise_list

    def next(self):
        torch.npu.current_stream().wait_stream(self.stream)
        next_input = self.next_input
        new_images = self.new_images
        noise_sigma = self.noise_sigma
        if next_input is not None:
            self.preload()       
        return next_input,new_images,noise_sigma