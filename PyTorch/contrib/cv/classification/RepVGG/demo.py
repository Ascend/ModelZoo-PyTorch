#!/usr/bin/env python3
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

import os
import torch
import numpy as np
from repvgg import get_RepVGG_func_by_name
from collections import OrderedDict

def proc_node_module(checkpoint, attr_name):
    new_model_state = OrderedDict()
    for k, v in checkpoint[attr_name].items():
        if(k[0: 7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_model_state[name] = v
    return new_model_state
    
    
def build_model():
    repvgg_build_func = get_RepVGG_func_by_name("RepVGG-A0")
    model = repvgg_build_func(deploy=False)
    checkpoint = torch.load("RepVGG-A0_hello_best.pth.tar")
    checkpoint["state_dict"] = proc_node_module(checkpoint, 'state_dict')
    model.load_state_dict(checkpoint["state_dict"])
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


if __name__ == '__main__':
    raw_data = get_raw_data()
    model = build_model()
    input_tensor = pre_process(raw_data)
    output_tensor = model(input_tensor)
    _, pred = output_tensor.topk(1, 1, True, True)
    print("class: ", pred[0][0].item())
