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
import models
from timm.models import create_model

def get_raw_data():
    from PIL import Image
    from urllib.request import urlretrieve
    with open('url.ini', 'r') as f:
        content = f.read()
        IMAGE_URL = content.split('img_url=')[1].split('\n')[0]
    urlretrieve(IMAGE_URL, 'tmp.jpg')
    img = Image.open("tmp.jpg")
    img = img.convert('RGB')
    return img

def test():
    loc = 'npu:0'
    loc_cpu = 'cpu'
    torch.npu.set_device(loc)


    model = create_model(
        'deit_small_patch16_224',
        pretrained=False,
        num_classes=1000,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
    )
    checkpoint = torch.load('test/output/checkpoint.pth', map_location=loc_cpu)
    model.load_state_dict(checkpoint['model'])
    model = model.to(loc)
    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    rd = get_raw_data()
    data_transfrom = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize])

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