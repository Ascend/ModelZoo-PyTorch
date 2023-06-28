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
import argparse
import torch
import torchvision
from torchvision import datasets, transforms
from collections import OrderedDict

from timm.models import create_model

parser = argparse.ArgumentParser(description='ImageNet Training')
parser.add_argument('--checkpoint', type=str, default='',
                    help='checkpoint path')
args = parser.parse_args()

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
    with open('url.ini', 'r') as f:
        content = f.read()
        IMAGE_URL = content.split('image_url=')[1].split('\n')[0]
    urlretrieve(IMAGE_URL, 'tmp.jpg')
    img = Image.open("tmp.jpg")
    img = img.convert('RGB')
    return img

def test():
    loc = 'npu:0'
    loc_cpu = 'cpu'
    torch.npu.set_device(loc)
    if args.checkpoint == '':
        print("please give the checkpoint path using --checkpoint param")
        exit(0)
    checkpoint = torch.load(args.checkpoint, map_location=loc)
    checkpoint['state_dict'] = proc_node_module(checkpoint, 'state_dict')
    
    model = create_model('vit_small_patch16_224', pretrained=False)
    
    model = model.to(loc)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    rd = get_raw_data()
    data_transfrom = transforms.Compose([
            transforms.RandomResizedCrop(224,interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    inputs = data_transfrom(rd)
    inputs = inputs.unsqueeze(0)
    inputs = inputs.to(loc)
    output = model(inputs)
    output = output.to(loc_cpu)

    _, pred = output.topk(1, 1, True, True)
    result = torch.argmax(output, 1)
    print("class: ", pred[0][0].item())
    print(result)

if __name__ == "__main__":
    test()