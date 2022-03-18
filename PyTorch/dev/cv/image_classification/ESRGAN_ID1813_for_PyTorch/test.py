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
import os
import cv2
from PIL import Image
from model.ESRGAN import ESRGAN
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from collections import OrderedDict
import argparse
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

parser = argparse.ArgumentParser()
#当前未使用pth文件，删除，后续使用再加
parser.add_argument('--gan_pth_path', default='parameters/gan.pth')
parser.add_argument('--psnr_pth_path', default='parameters/psnr.pth')
parser.add_argument('--interp_pth_path', default='parameters/interp.pth')
parser.add_argument('--lr_dir')
parser.add_argument('--sr_dir')
parser.add_argument('--alpha', type=int, default=0.8)

args = parser.parse_args()


net_PSNR = torch.load(args.psnr_pth_path)
net_ESRGAN = torch.load(args.gan_pth_path)
net_interp = OrderedDict()

for k, v_PSNR in net_PSNR.items():
    v_ESRGAN = net_ESRGAN[k]
    net_interp[k] = (1 - args.alpha) * v_PSNR + args.alpha * v_ESRGAN

if not os.path.exists(args.lr_dir):
    raise Exception('[!] No lr path')
if not os.path.exists(args.sr_dir):
    os.makedirs(args.sr_dir)

device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')

with torch.no_grad():
    net = ESRGAN(3, 3, scale_factor=4)
    net.load_state_dict(net_interp)
    net = net.to(f'npu:{NPU_CALCULATE_DEVICE}').eval()

    for image_name in os.listdir(args.lr_dir):
        image = Image.open(os.path.join(args.lr_dir, image_name)).convert('RGB')
        image = TF.to_tensor(image).to(f'npu:{NPU_CALCULATE_DEVICE}').unsqueeze(dim=0)

        image = net(image)

        save_image(image,  os.path.join(args.sr_dir, image_name))
        print(f'save {image_name}')
