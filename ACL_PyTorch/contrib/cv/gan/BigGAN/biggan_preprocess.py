# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
This script randomly generates 'noise' and 'label' as the input of model.
"""

import os
import sys
import torch
import argparse
import numpy as np

from BigGAN import Generator
from torch.nn import Embedding
from collections import OrderedDict

##############################################################################
# Class
##############################################################################
class Distribution(torch.Tensor):
    def init_distribution(self, dist_type, **kwargs):
        self.dist_type = dist_type
        self.dist_kwargs = kwargs
        if self.dist_type == 'normal':
            self.mean, self.var = kwargs['mean'], kwargs['var']
        elif self.dist_type == 'categorical':
            self.num_categories = kwargs['num_categories']

    def sample_(self):
        if self.dist_type == 'normal':
            self.normal_(self.mean, self.var)
        elif self.dist_type == 'categorical':
            self.random_(0, self.num_categories)

    def to(self, *args, **kwargs):
        new_obj = Distribution(self)
        new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
        new_obj.data = super().to(*args, **kwargs)
        return new_obj


##############################################################################
# Functions
##############################################################################
def proc_nodes_module(checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if (k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def prepare_noise_label(embedding, batch_size, dim_z=120, nclasses=1000, device='cpu', z_var=1.0):
    z_ = Distribution(torch.randn(batch_size, dim_z, requires_grad=False))
    z_.init_distribution('normal', mean=0, var=z_var)
    z_.sample_()
    noise = z_.to(device, torch.float32).detach().numpy()
    noise_list = np.split(noise, 6, 1)
    noise = noise_list[0]
    noise = np.expand_dims(noise, axis=1)

    y_ = Distribution(torch.zeros(batch_size, requires_grad=False))
    y_.init_distribution('categorical', num_categories=nclasses)
    y_.sample_()
    label = y_.to(device, torch.int64)
    label = embedding(label).detach().numpy()
    label = [np.concatenate([label, item], 1) for item in noise_list[1:]]
    label = np.array(label)
    label = label.transpose((1, 0, 2))

    return noise, label, y_


def input_preprocess(embedding, args):
    noise_path = args.prep_noise
    noise_path = os.path.realpath(noise_path)
    if not os.path.exists(noise_path):
        os.makedirs(noise_path)

    label_path = args.prep_label
    label_path = os.path.realpath(label_path)
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    y = []
    for i in range(int(np.ceil(args.num_inputs / args.batch_size))):
        if i % 1000 == 0:
            print("has generated input pair {:05d}...".format(i*args.batch_size))

        noise, label, y_ = prepare_noise_label(embedding=embedding, batch_size=args.batch_size)
        noise.tofile(os.path.join(noise_path, "input_{:05d}.bin".format(i)))
        label.tofile(os.path.join(label_path, "input_{:05d}.bin".format(i)))
        y += [y_.cpu().numpy()]

    y = np.concatenate(y, 0)[:args.num_inputs]
    y_npz_filename = 'gen_y' + '.npz'
    np.savez(y_npz_filename, **{'y': y})


##############################################################################
# Main
##############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', type=str, default="./G_ema.pth")
    parser.add_argument('--prep-noise', type=str, default="./prep_noise")
    parser.add_argument('--prep-label', type=str, default="./prep_label")
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-inputs', type=int, default=50000,
                        help="Number of noise and label as the input of model")
    opt = parser.parse_args()

    # load checkpoint
    checkpoint = torch.load(opt.pth, map_location=torch.device('cpu'))
    checkpoint = proc_nodes_module(checkpoint)
    embedding_layer = Embedding(num_embeddings=1000, embedding_dim=128,
                                _weight=checkpoint['shared.weight'])

    input_preprocess(embedding_layer, opt)
