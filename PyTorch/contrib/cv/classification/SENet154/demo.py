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

# -*- coding: utf-8 -*-
"""demo.py
"""
import os
import sys
import torch


sys.path.append('.')

import senet 


device = "cpu"


def build_model(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("File '{}' not found!".format(file_path))

    model = senet.senet154(num_classes=1000, pretrained='imagenet', use_pretrained=False).to(device)
    state_dict = torch.load(file_path, map_location=device)['net']
    model.load_state_dict({k.replace('module.', '', 1): v for k, v in state_dict.items()})
    model.eval()

    return model


def get_raw_data():
    from PIL import Image
    from urllib.request import urlretrieve
    cur_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(cur_path, 'url.ini'), 'r') as f:
        content = f.read()
        img_url = content.split('img_url=')[1].split('\n')[0]
    IMAGE_URL = img_url
    urlretrieve(IMAGE_URL, 'tmp.jpg')
    img = Image.open("tmp.jpg")
    img = img.convert('RGB')
    return img


def pre_process(raw_data):
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
    return torch.argmax(output_tensor, 1)


if __name__ == '__main__':
    # 1. 获取原始数据
    raw_data = get_raw_data()
    # 2. 构建模型
    model = build_model(sys.argv[1])
    # 3. 预处理
    input_tensor = pre_process(raw_data)
    # 4. 执行forward
    output_tensor = model(input_tensor)
    # 5. 后处理
    result = post_process(output_tensor)
    # 6. 打印
    print(result)
