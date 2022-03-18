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
from PIL import Image
from lib.PraNet_Res2Net import PraNet
import torch.nn.functional as F
import numpy as np


class test_dataset:
    # path = './data/TestDataset'
    def __init__(self, path):
        self.testsize = 352
        self.data_path = '{}/images/cju0u82z3cuma0835wlxrnrjv.png'.format(path)
        self.gt_path = '{}/masks/cju0u82z3cuma0835wlxrnrjv.png'.format(path)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()

    def load_data(self):
        image = self.rgb_loader(self.data_path)
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gt_path)
        return image, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


    
def test():
    path = './data/TestDataset/Kvasir'
    loc = 'npu:0'
    loc_cpu = 'cpu'
    torch.npu.set_device(loc)
    model = PraNet()
    pretrained_dict = torch.load("./snapshots/PraNet_Res2Net/PraNet-19.pth", map_location="cpu")
    model.load_state_dict({k.replace('module.',''):v for k, v in pretrained_dict.items()})
    if "fc.weight" in pretrained_dict:
        pretrained_dict.pop('fc.weight')
        pretrained_dict.pop('fc.bias')
    model.load_state_dict(pretrained_dict, strict=False)
    model = model.to(loc)
    model.eval()

    test_loader = test_dataset(path)
    image, gt = test_loader.load_data()

    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)

    image = image.npu()

    res5, res4, res3, res2 = model(image)
    res = res2
    res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    print('res is ', res)
if __name__ == "__main__":
    test()