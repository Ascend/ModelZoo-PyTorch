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

import os

import torch
import numpy as np
from inceptionresnetv2 import InceptionResNetV2
import argparse
from apex import amp
import apex
import torch.distributed as dist
parser = argparse.ArgumentParser(description='inceptionresnetv2 demo')
parser.add_argument('--device', default='npu', type=str,
                    help='npu or gpu')

parser.add_argument('--device-list', default='0,1,2,3,4,5,6,7', type=str, help='device id list')
parser.add_argument('--dist-backend', default='hccl', type=str,
                    help='distributed backend')
parser.add_argument('--addr', default='127.0.0.1', type=str,
                    help='master addr')

def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map


def build_model():
    global loc
    args = parser.parse_args()
    args.process_device_map = device_id_to_process_device_map(args.device_list)
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '29688'
    ngpus_per_node = len(args.process_device_map)
    
    dist.init_process_group(backend=args.dist_backend,  
                                    world_size=1, rank=0)
    
    args.gpu = args.process_device_map[0]
    loc = 'npu:{}'.format(args.gpu)
    torch.npu.set_device(loc)
    
    model = InceptionResNetV2().to(loc)
    optimizer = apex.optimizers.NpuFusedSGD(model.parameters(), 0.8,
                                momentum=0.9,
                                weight_decay=1.0e-04)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=1024)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False)
    checkpoint = torch.load('checkpoint.pth.tar')
    model.load_state_dict(checkpoint['state_dict'],False)
    model.eval()  
    return model


def get_raw_data():
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
    from torchvision import transforms
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    transforms_list = transforms.Compose([
        transforms.Resize(342),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        normalize
    ])
    input_data = transforms_list(raw_data)
    return input_data.unsqueeze(0)


def post_process(output_tensor):
    print(output_tensor)
    return torch.argmax(output_tensor, 1)


if __name__ == '__main__':
    raw_data = get_raw_data()
    model = build_model()
    output_tensor = model(input_tensor.to(loc))
    result = post_process(output_tensor)

    print(result)

