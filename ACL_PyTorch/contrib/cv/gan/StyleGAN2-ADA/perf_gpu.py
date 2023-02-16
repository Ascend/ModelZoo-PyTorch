# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

import os
import sys
import time

sys.path.append('./model')
import numpy as np
import torch
import pickle
import argparse
import PIL.Image
import functools

from tqdm import tqdm


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def main(args):
    pkl_file = args.pkl_file
    bs = args.batch_size
    input_path = args.input_path
    image_path = args.image_path
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    grid_size = (1, 1)
    input_files = os.listdir(input_path)
    input_files.sort()
    image_path = os.path.join(image_path, 'pkl_img')
    os.makedirs(image_path, exist_ok=True)
    # load model
    start = time.time()
    with open(pkl_file, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)

    G.forward = functools.partial(G.forward, force_fp32=True)
    for i in tqdm(range(len(input_files))):
        input_file = input_files[i]
        input_file = os.path.join(input_path, input_file)
        input_file = np.fromfile(input_file, dtype=np.float32)
        z = torch.tensor(input_file).reshape(-1, G.z_dim).to(device)
        c = torch.empty(bs, 0).to(device)
        image = G(z, c)
        image = image.reshape(-1, 3, 512, 512)
        image = image.cpu()
        save_image_grid(image, os.path.join(image_path, f'gen_image_{i:04d}') + '.png', drange=[-1, 1],
                        grid_size=grid_size)

    end = time.time()
    print(f'Inference average time : {((end - start) * 1000 / len(input_files)):.2f} ms')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_file', type=str, default='./G_ema_bs8_8p_kimg1000.pkl')
    parser.add_argument('--input_path', type=str, default='./pre_data')
    parser.add_argument('--image_path', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    main(args)
