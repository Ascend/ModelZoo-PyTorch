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


import os

import torch
import numpy as np
from timm.models import create_model
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser(description='xception demo ')
parser.add_argument('--device', default='npu', type=str,
                    help='npu or gpu')

parser.add_argument('--device-list', default='0,1,2,3,4,5,6,7', type=str, help='device id list')
parser.add_argument('--dist-backend', default='hccl', type=str,
                    help='distributed backend')
parser.add_argument('--addr', default='192.168.88.3', type=str,
                    help='master addr')

parser.add_argument('--model-path', default='', type=str, metavar='PATH',
                    help='mode parh')

def proc_node_module(checkpoint, attr_name):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[attr_name].items():
        if(k[0: 7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict

def build_model():
    args = parser.parse_args()
    global loc
    loc = 'npu:0'
    torch.npu.set_device(loc)
    model = create_model(
        'cspresnext50',
        num_classes=1000
        )
    model_path = args.model_path
    checkpoint = torch.load(model_path,  map_location=loc)
    checkpoint["state_dict"] = proc_node_module(checkpoint, "state_dict")

    model = model.to(loc)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model    

def get_raw_data():
    # 请自定义获取数据方式，请勿将原始数据上传至代码仓
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
    # 请自定义模型预处理方法
    from torchvision import transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transforms_list = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(244),
        transforms.ToTensor(),
        normalize
    ])
    input_data = transforms_list(raw_data)
    return input_data.unsqueeze(0)


def post_process(output_tensor):
    # 请自定义后处理方法
    print(output_tensor)
    return torch.argmax(output_tensor, 1)


if __name__ == '__main__':
    # 1. 获取原始数据
    raw_data = get_raw_data()

    # 2. 构建模型
    model = build_model()

    # 3. 预处理
    input_tensor = pre_process(raw_data)

    # 4. 执行forward
    output_tensor = model(input_tensor.to(loc))

    # 5. 后处理
    result = post_process(output_tensor)

    # 6. 打印
    print(result)
