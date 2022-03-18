#!/usr/bin/env python
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import torch
import torchvision
from torchvision import datasets, transforms
from ghostnet.ghostnet_pytorch.ghostnet import ghostnet

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--data', metavar='DIR', default='/opt/npu/imagenet',
                    help='path to image folder')
                    
def test(args):
    loc = 'cpu'
    checkpoint = torch.load("model_best.pth.tar", map_location=loc)
    model = ghostnet().to(loc)
    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    data_transfrom = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize])
    
    img = datasets.ImageFolder(args.data, transform=data_transfrom)

    imgLoader = torch.utils.data.DataLoader(img, batch_size=1, shuffle=False, num_workers=1)

    inputs, _ = next(iter(imgLoader))
    inputs = inputs.to(loc)
    output = model(inputs)
    

    _, pred = output.topk(1, 1, True, True)
    print("class: ", pred[0][0].item())

if __name__ == "__main__":
    args = parser.parse_args()
    test(args)

