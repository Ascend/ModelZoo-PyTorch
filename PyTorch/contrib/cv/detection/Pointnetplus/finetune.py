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

import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm
import models.provider
from models.ModelNetDataLoader import ModelNetDataLoader
import models.pointnet2_cls_ssg as models

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True
        
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=2, type=int, help='number of epoch in training')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=True, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--device', type=str,default='cpu',help='which device to use')
    parser.add_argument('--data',type=str, default='./modelnet40_normal_resampled', help='data_path')
    parser.add_argument('--num_class',type=int,default=41,help='num of class')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--model_pth', type=str, default='./log/classification/pointnet2_cls_ssg/checkpoints/', help='Point Number')
    parser.add_argument('--worker', type=int, default=1, help='number ofs workers')

    return parser.parse_args()

def main(args):
    test_dataset = ModelNetDataLoader(root=args.data, args=args, split='test', process_data=args.process_data)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.worker)
    model_dict = torch.load(str(args.model_pth)+'best_model.pth', map_location='cpu')['model_state_dict']
    model_dict.pop('fc3.weight')
    model_dict.pop('fc3.bias')
    classifier = models.get_model(args.num_class, normal_channel=args.use_normals)
    classifier.apply(inplace_relu)
    if args.device !='cpu':
        classifier = classifier.npu()
    classifier.load_state_dict(model_dict, strict=False)

    for epoch in range(args.epoch):
        for batch_id,(points, target) in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader)):
            points = points.transpose(2, 1)
            if args.device !='cpu':
                points, target = points.npu(), target.npu()
            pred, trans_feat = classifier(points)
            pred_choice = pred.data.max(1)[1]
            print("output class is",pred_choice)

if __name__ == '__main__':
    args = parse_args()
    main(args)


