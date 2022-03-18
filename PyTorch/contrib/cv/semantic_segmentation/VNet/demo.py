# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import argparse
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.multiprocessing import spawn
import torchvision.transforms as transforms

import torchbiomed.datasets as dset
import torchbiomed.loss as bioloss

import os
import shutil
import cv2

import vnet
from apex import amp
import apex

nodule_masks = "normalized_nodule_mask"
lung_masks = "normalized_lung_mask"
ct_images = "normalized_lung_ct"
ct_targets = lung_masks
target_split = [2, 2, 2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--device_num', type=int, default=1)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--opt_level', type=str, default='O2')
    parser.add_argument('--data', type=str, default='/opt/npu/dataset/luna16')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--save', type=str)
    args = parser.parse_args()
    args.device_id = args.device+':0'
    model = vnet.VNet(elu=False, nll=True)
    model = model.to(args.device_id)
    if args.amp:
        model = amp.initialize(model, opt_level=args.opt_level)
    
    if os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=args.device_id)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        return

    normMu = [-642.794]
    normSigma = [459.512]
    normTransform = transforms.Normalize(normMu, normSigma)
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])

    testSet = dset.LUNA16(root=args.data, images=ct_images, targets=ct_targets,
                    mode="test", transform=testTransform, masks=None, split=target_split)
    testLoader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=4, sampler=None)
    
    model.eval()
    incorrect = 0
    numel = 0
    with torch.no_grad():
        for data, target in testLoader:
            data, target = data.to(args.device_id), target.to(args.device_id)
            target = target.view(target.numel())
            numel += target.numel()
            output = model(data)
            output = output.view(-1,2)
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            incorrect += pred.ne(target.data).cpu().sum().item()
            break
    err = 100.*incorrect/numel
    print('Error rate: {}/{} ({:.4f}%)\n'.format(incorrect, numel, err))
    pred = pred.view(64,80,80).cpu().numpy()
    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)
    for i in range(len(pred)):
        img = pred[i]*255
        cv2.imwrite(os.path.join(args.save,str(i)+'.png'),img)

if __name__ == '__main__':
    main()
