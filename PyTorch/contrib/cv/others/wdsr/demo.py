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
import importlib
import os

from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms


def get_raw_data(params):
    path = params.lr_image
    lr_image = Image.open(path)
    lr_image = np.ascontiguousarray(lr_image)
    lr_image = transforms.functional.to_tensor(lr_image)

    lr_image = torch.reshape(lr_image, (1, lr_image.shape[0], lr_image.shape[1], lr_image.shape[2]))

    return lr_image


def test(parser):
    params = parser.parse_args()
    loc = 'npu:0'
    loc_cpu = 'cpu'
    # torch.npu.set_device(loc)
    checkpoint = torch.load(params.pre_train_model, loc)
    model_module = importlib.import_module('models.wdsr')
    model_module.update_argparser(parser)
    params = parser.parse_args()
    model, criterion, optimizer, lr_scheduler = model_module.get_model_spec(params)
    model = model.to(loc)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    inputs = get_raw_data(params)
    inputs = inputs.to(loc, non_blocking=True)
    output = model(inputs)
    output = output.to(loc_cpu)
    output = output.detach().numpy()
    output = output.reshape(output.shape[1], output.shape[2], output.shape[3])
    output = output.transpose((1, 2, 0))
    outImage = Image.fromarray((output * 255.).astype('uint8')).convert('RGB')
    outName = params.lr_image.split("/")[-1]
    if not os.path.exists(params.save):
        os.makedirs(params.save)
    outImage.save(params.save+outName)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pre_train_model',
        help='File path to load checkpoint.',
        default=None,
        type=str,
    )
    parser.add_argument(
        '--lr_image',
        help='input image path.',
        default=None,
        type=str,
    )
    parser.set_defaults(
        dataset="div2k",
        image_mean=0.5,
        num_channels=3,
        scale=2,
        save="output_sr/"
    )
    test(parser)
