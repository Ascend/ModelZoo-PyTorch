# Copyright 2020 Huawei Technologies Co., Ltd
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
from vgg import vgg19
from collections import OrderedDict

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--data', metavar='DIR', default='/dataset/imagenet',
                    help='path to image folder')

def proc_node_module(checkpoint, attr_name):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[attr_name].items():
        if(k[0: 7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict

def test(args):
    loc = 'cpu'
    checkpoint = torch.load("./checkpoint.pth.tar", map_location=loc)
    checkpoint['state_dict'] = proc_node_module(checkpoint, 'state_dict')
    model = vgg19().to(loc)
    model.load_state_dict(checkpoint['state_dict'])
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