# -*- coding: utf-8 -*-
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
"""demo.py
"""

import torch
import torch.npu
loc = 'npu:0'

def build_model():
    from network import ShuffleNetV2_Plus
    architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
    md = ShuffleNetV2_Plus(architecture=architecture, model_size='Small')
    md = md.to(loc)
    md.eval()
    pretrained = torch.load('trainedmodel.pth.tar', map_location=loc) # change this to the filename of the trained model!

    old_dict = pretrained['state_dict']
    state_dict = {}
    for key, value in old_dict.items():
        key = key[7:]
        state_dict[key] = value

    md.load_state_dict(state_dict)
    return md


def get_raw_data():
    from PIL import Image
    from urllib.request import urlretrieve
    IMAGE_URL = 'https://bbs-img.huaweicloud.com/blogs/img/thumb/1591951315139_8989_1363.png'
    urlretrieve(IMAGE_URL, 'tmp.jpg')
    img = Image.open("tmp.jpg")
    img = img.convert('RGB')
    return img


def pre_process(rd):

    from torchvision import transforms
    transforms_list = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    input_data = transforms_list(rd)
    return input_data.unsqueeze(0)


def post_process(out):
    return torch.argmax(out, 1)


if __name__ == '__main__':
    torch.npu.set_device(loc)
    # 1. 获取原始数据
    raw_data = get_raw_data()

    # 2. 构建模型
    model = build_model()

    # 3. 预处理
    input_tensor = pre_process(raw_data)
    input_tensor = input_tensor.to(loc)
    # 4. 执行forward
    output_tensor = model(input_tensor)
    output_tensor = output_tensor.cpu()
    # 5. 后处理
    result = post_process(output_tensor)
    # 6. 打印
    print(result)
