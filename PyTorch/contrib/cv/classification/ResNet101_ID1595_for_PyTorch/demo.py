# -*- coding: utf-8 -*-
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

import os
import torch
import numpy as np


def build_model():
    import torchvision
    model = torchvision.models.resnet101(pretrained=True)
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
    # 1. get raw data
    raw_data = get_raw_data()

    # 2. buid model
    model = build_model()

    # 3. pre process data
    input_tensor = pre_process(raw_data)

    # 4. run forward
    output_tensor = model(input_tensor)

    # 5. post process
    result = post_process(output_tensor)

    # 6. print result
    print(result)