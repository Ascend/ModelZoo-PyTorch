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
import six
import sys
sys.path.append('../../')
import collections
import cv2
import time
from models.resnet import rf_lw50
from utils.helpers import prepare_img
import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

def create_visual_anno(anno):
    """"""
    assert np.max(anno) <= 10, "only 10 classes are supported, add new color in label2color_dict"
    label2color_dict = {
        0: [0, 0, 0],
        1: [255, 248, 220],  # cornsilk
        2: [100, 149, 237],  # cornflowerblue
        3: [102, 205, 170],  # mediumAquamarine
        4: [205, 133, 63],  # peru
        5: [160, 32, 240],  # purple
        6: [255, 64, 64],  # brown1
        7: [139, 69, 19],  # Chocolate4
        8: [255,0,0],
        9: [0,255,0],
        10:[0,0,255]
    }
    # visualize
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno


has_cuda = torch.npu.is_available()
n_classes = 11
net = rf_lw50(n_classes, imagenet=False, pretrained=False)
cpkt = torch.load("../face/checkpoint.pth.tar")['segmenter']
weights = collections.OrderedDict()
for key in cpkt:
    print(key.split('.',1))
    weights[key.split('.',1)[1]] = cpkt[key]

net.load_state_dict(weights)
net = net.npu()
net.eval()
img_path = "/home/kong/Downloads/d94be52120f2aa2cfbd7c12f10817b04.jpeg"
with torch.no_grad():

    img = np.array(Image.open(img_path))
    img = cv2.resize(img,(512,512))
    orig_size = img.shape[:2][::-1]

    img_inp = torch.tensor(prepare_img(img).transpose(2, 0, 1)[None]).float()
    if has_cuda:
        img_inp = img_inp.npu()

    plt.imshow(img)
    start = time.time()

    segm = net(img_inp)[0].data.cpu().numpy().transpose(1, 2, 0)
    end = time.time()
    segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC)
    segm = segm.argmax(axis=2).astype(np.uint8)
    print("Infer time :",end-start)

segm_rgb = create_visual_anno(segm)
image_add = cv2.addWeighted(img,0.8,segm_rgb,0.2,0)
result = np.hstack((img,segm_rgb,image_add))
result = Image.fromarray(result.astype(np.uint8))
result.save("face_seg3.jpg")
