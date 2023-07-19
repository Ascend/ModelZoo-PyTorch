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

# -*- coding: utf-8 -*-
"""demo.py
"""

import torch
import numpy as np
from models.nasnet_mobile import nasnetamobile

def build_model():
    # Creat and load model
    model = nasnetamobile(num_classes=1000, pretrained='imagenet')
    model.eval()
    return model


def get_raw_data():
    # define the way to get data
    from PIL import Image
    from urllib.request import urlretrieve
    with open('url.ini', 'r') as f:
        content = f.read()
        img_url = content.split('img_url=')[1].split('\n')[0]
    IMAGE_URL = img_url
    urlretrieve(IMAGE_URL, 'tmp.jpg')
    img = Image.open("tmp.jpg")
    img = img.convert('RGB')
    return img


def pre_process(raw_data):
    # the method for pre_process
    from torchvision import transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transforms_list = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    input_data = transforms_list(raw_data)
    return input_data.unsqueeze(0)


def post_process(output_tensor):
    # method od post_process
    return torch.argmax(output_tensor, 1)


if __name__ == '__main__':
    # get data
    raw_data = get_raw_data()

    # creat model
    model = build_model()

    # pre_process
    input_tensor = pre_process(raw_data)

    # forward
    output_tensor = model(input_tensor)

    # post_process
    result = post_process(output_tensor)

    # print
    print(result)