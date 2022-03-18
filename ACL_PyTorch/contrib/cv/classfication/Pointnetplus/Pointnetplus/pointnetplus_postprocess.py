# Copyright 2021 Huawei Technologies Co., Ltd
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

import argparse
import numpy as np
import os
import torch
import logging
import sys
from tqdm import tqdm


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('off_line_pred')
    parser.add_argument('--target_path', type=str, default='out/bs1/',
                        required=False, help='target root')
    parser.add_argument('--data_loc', type=str, default='/home/data/modelnet40_normal_resampled/', required=False, help='data location')
    parser.add_argument('--batch_size', type=int, default=1, required=False,
                        help='batch size')
    args = parser.parse_args()
    out_folder = os.listdir(args.target_path)[-1]
    args.target_path = os.path.join(args.target_path, out_folder)
    return args

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def model_data_loader(data_pth):
    data_path = data_pth
    catfile = os.path.join(data_path, 'modelnet40_shape_names.txt')
    cat = [line.rstrip() for line in open(catfile)]
    classes = dict(zip(cat, range(len(cat))))
    shape_ids = {}
    shape_ids['test'] = [line.rstrip() for line in open(os.path.join(data_path, 'modelnet40_test.txt'))]
    split = 'test'
    shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
    datapath = [(shape_names[i], os.path.join(data_path, shape_names[i], shape_ids[split][i]) + '.txt') for i
                        in range(len(shape_ids[split]))]
    print('The size of %s data is %d' % (split, len(datapath)))
    return classes, datapath

def test(pred_path,data_path):
    mean_correct = []
    class_acc = np.zeros((40, 3))

    classes,data_pth = model_data_loader(data_path)
    print('data is %d' % len(data_pth))
    # load infer results
    def load_infer_results():
        num_out = len(data_pth) // args.batch_size
        for j in range(num_out):
            pred_loca = os.path.join(pred_path, 'part2_' + str(j) + '_output_0.bin')
            pred = np.fromfile(pred_loca,np.float32)
            if args.batch_size == 1:
                pred.shape = 1, 40
                pred = torch.from_numpy(pred)
                yield pred
            else:
                pred.shape = args.batch_size, 40
                for d in pred:
                    d = torch.from_numpy(np.expand_dims(d, axis=0))
                    yield d
    infer_results = load_infer_results()

    # load gt results
    num_results = len(data_pth) // args.batch_size * args.batch_size
    for j in tqdm(range(num_results)):
        fn = data_pth[j]
        cls = classes[data_pth[j][0]]
        target = np.array([cls]).astype(np.int32)
        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
        point_set = point_set[0:1024, :]
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        point_set = point_set[:, 0:3]
        point_set = point_set[None,]
        new_point_set = torch.from_numpy(point_set)
        points = new_point_set.transpose(2, 1)
        target = torch.from_numpy(target)

        pred = next(infer_results)
        pred_choice = pred.data.max(1)[1]
        '''验证精度'''
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    print('class_acc is %f' % class_acc)
    print('instance_acc is %f' % instance_acc)
    return instance_acc, class_acc

if __name__ == '__main__':
    args = parse_args()
    test(args.target_path, args.data_loc)
