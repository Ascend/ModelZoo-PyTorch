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
if torch.__version__>= '1.8':
    import torch_npu
import torchvision
from torchvision import datasets, transforms
from collections import OrderedDict
import sys
ckpt_pth = sys.argv[1]

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

def test():
    loc = 'npu:0'
    loc_cpu = 'cpu'
    torch.npu.set_device(loc)
    import timm
    print("start load model")
    model = timm.create_model('tresnet_m', checkpoint_path=ckpt_pth).npu()
    model.eval()

    normalize = transforms.Normalize(mean=[0, 0, 0],
                                    std=[1, 1, 1])
    rd = get_raw_data()
    data_transfrom = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize])

    inputs = data_transfrom(rd)
    inputs = inputs.unsqueeze(0)
    inputs = inputs.to(loc)
    print("start classification")
    output = model(inputs)
    output = output.to(loc_cpu)

    _, pred = output.topk(1, 1, True, True)
    result = torch.argmax(output, 1)
    print("class: ", pred[0][0].item())
    print(result)

if __name__ == "__main__":
    test()