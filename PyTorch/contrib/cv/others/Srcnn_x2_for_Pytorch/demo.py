# -*- coding: utf-8 -*-
#
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

import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb

import torch.npu
from skimage.metrics import peak_signal_noise_ratio

from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre-train-path', type=str, default="test/npu_1p/x2/best.pth")
    parser.add_argument('--image-file', type=str, default="data/ppt3.bmp")
    parser.add_argument('--scale', type=int, default=2)
    args = parser.parse_args()
    return args

def test():
    loc = 'npu:0'
    loc_cpu = 'cpu'
    torch.npu.set_device(loc)

    model = SRCNN()
    model = model.to(loc)
    
    checkpoint = torch.load(args.pre_train_path, map_location='cpu')
    if 'module.' in list(checkpoint['model'].keys())[0]:
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(checkpoint['model'], strict=False)

    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
    image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)
    image.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(loc)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    y = y.cpu().numpy()
    preds = preds.cpu().numpy()
    psnr = peak_signal_noise_ratio(y, preds)
    print('PSNR: {:.2f}'.format(psnr))

    preds = torch.from_numpy(preds).to(loc)
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(args.image_file.replace('.', '_srcnn_x{}.'.format(args.scale)))

if __name__ == "__main__":
    
    args = parse_args()
    test()