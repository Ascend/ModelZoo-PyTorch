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
import argparse
import torch
import torchvision
from torchvision import datasets, transforms
import pnasnet as pnasnet
from collections import OrderedDict

def proc_node_module(checkpoint, attr_name):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[attr_name].items():
        if(k[0: 7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict

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

def test():
    loc = 'npu:0'
    loc_cpu = 'cpu'
    torch.npu.set_device(loc)
    checkpoint = torch.load("./test/checkpoint.pth.tar", map_location=loc)
    checkpoint['state_dict'] = proc_node_module(checkpoint, 'state_dict')
    model = pnasnet.pnasnet5large(num_classes=1000, pretrained=False,use_aux=True)
    model = model.to(loc)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.229, 0.5, 0.5])
    rd = get_raw_data()
    data_transfrom = transforms.Compose([
                transforms.Resize(int(331/0.875)),
                transforms.CenterCrop(331),
                transforms.ToTensor(),
                normalize])

    inputs = data_transfrom(rd)
    inputs = inputs.unsqueeze(0)
    inputs = inputs.to(loc)
    output, _ = model(inputs)
    output = output.to(loc_cpu)

    _, pred = output.topk(1, 1, True, True)
    result = torch.argmax(output, 1)
    print("class: ", pred[0][0].item())
    print(result)

if __name__ == "__main__":
    test()