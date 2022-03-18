# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import models
import os
import numpy as np
from data_loader import BSDS_RCFLoader
from torch.utils.data import DataLoader
from PIL import Image
import scipy.io as io
import argparse

CALCULATE_DEVICE = "npu:0"

parser = argparse.ArgumentParser(description='RCF')
parser.add_argument('--resume', help='ckpt/only-final-lr-0.008-iter-50000.pth', type=str)
args = parser.parse_args()

# ckpt
# resume = 'ckpt/only-final-lr-0.008-iter-50000.pth'
resume = args.resume

folder = 'results/val/'
all_folder = os.path.join(folder, 'all')
png_folder = os.path.join(folder, 'png')
mat_folder = os.path.join(folder, 'mat')
batch_size = 1
assert batch_size == 1

try:
    os.mkdir(all_folder)
    os.mkdir(png_folder)
    os.mkdir(mat_folder)
except Exception:
    print('dir already exist....')
    pass

model = models.resnet101(pretrained=False).npu()
model.eval()

#resume..
checkpoint = torch.load(resume)
model.load_state_dict(checkpoint)

test_dataset = BSDS_RCFLoader(split="test")
test_loader = DataLoader(
    test_dataset, batch_size=batch_size,
    num_workers=1, drop_last=True, shuffle=False)

torch.npu.set_device(CALCULATE_DEVICE)

with torch.no_grad():
    for i, (image, ori, img_files) in enumerate(test_loader):
        h, w = ori.size()[2:]
        image = image.npu()
        name = img_files[0][5:-4]

        outs = model(image, (h, w))
        fuse = outs[-1].squeeze().detach().cpu().numpy()

        outs.append(ori)

        idx = 0
        print('working on .. {}'.format(i))

        for result in outs:
            idx += 1
            result = result.squeeze().detach().cpu().numpy()
            if len(result.shape) == 3:
                result = result.transpose(1, 2, 0).astype(np.uint8)
                result = result[:, :, [2, 1, 0]]
                Image.fromarray(result).save(os.path.join(all_folder, '{}-img.jpg'.format(name)))
            else:
                result = (result * 255).astype(np.uint8)
                Image.fromarray(result).save(os.path.join(all_folder, '{}-{}.png'.format(name, idx)))
        Image.fromarray((fuse * 255).astype(np.uint8)).save(os.path.join(png_folder, '{}.png'.format(name)))
        io.savemat(os.path.join(mat_folder, '{}.mat'.format(name)), {'result': fuse})
    print('finished.')

