# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import argparse
import os
import datetime
import numpy as np
import torch.npu
import torch
import torch.nn as nn
from pdb import set_trace as st
from PIL import Image
from torchvision import transforms
import deeplab
from data_loader import CelebASegmentation

# resnet_file_spec = dict(file_url='https://drive.google.com/uc?id=1oRGgrI4KNdefbWVpw0rRkEP1gbJIRokM', file_path='deeplab_model/R-101-GN-WS.pth.tar', file_size=178260167, file_md5='aa48cc3d3ba3b7ac357c1489b169eb32')
# deeplab_file_spec = dict(file_url='https://drive.google.com/uc?id=1w2XjDywFr2NjuUWaLQDRktH7VwIfuNlY', file_path='deeplab_model/deeplab_model.pth', file_size=464446305, file_md5='8e8345b1b9d95e02780f9bed76cc0293')

resolution = 128
model_fname = 'deeplab_model/deeplab_model.pth'
dataset_root = "ffhq_aging128x128"

assert torch.npu.is_available()
assert os.path.isdir(dataset_root)

dataset = CelebASegmentation(dataset_root, crop_size=513)
print("len(dataset)", len(dataset))
print("dataset.CLASSES", dataset.CLASSES)
print("dataset.images[0]", dataset.images[0])
print("Start time:", datetime.datetime.now())

model = getattr(deeplab, 'resnet101')(
    pretrained=True,
    num_classes=len(dataset.CLASSES),
    num_groups=32,
    weight_std=True,
    beta=False)

checkpoint = torch.load(model_fname,  map_location='cpu')
state_dict = {k[7:]: v for k,
              v in checkpoint['state_dict'].items() if 'tracked' not in k}
model.load_state_dict(state_dict)

device = "npu"
# model = model.npu()
model = model.to(device)
model.eval()

for i in range(len(dataset)):
    inputs = dataset[i]
    # inputs = inputs.npu()
    inputs = inputs.unsqueeze(0).to(device)
    # print("inputs-----",inputs.shape)
    outputs = model(inputs)
    # print("outputs-----",outputs.shape)

    _, pred = torch.max(outputs, 1)
    pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
    imname = os.path.basename(dataset.images[i])
    mask_pred = Image.fromarray(pred)
    mask_pred = mask_pred.resize((resolution, resolution), Image.NEAREST)
    try:
        mask_pred.save(dataset.images[i].replace(
            imname, 'parsings/'+imname[:-4]+'.png'))
    except FileNotFoundError:
        os.makedirs(os.path.join(os.path.dirname(
            dataset.images[i]), 'parsings'))
        mask_pred.save(dataset.images[i].replace(
            imname, 'parsings/'+imname[:-4]+'.png'))

    print('processed {0}/{1} images, Time:{2}'.format(i +
          1, len(dataset), datetime.datetime.now()))
    # break
