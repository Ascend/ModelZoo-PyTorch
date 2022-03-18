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

import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
sys.path.append(r"./Efficient-3DCNNs")
from dataset import get_test_set
from spatial_transforms import Normalize, Compose, Scale, CornerCrop
from temporal_transforms import TemporalRandomCrop
from target_transforms import VideoID

class ToTensor(object):
    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        img = np.array(pic, dtype=np.float32)
        img = img.transpose(2, 0, 1)
        img = img / self.norm_value
        return torch.from_numpy(img)

    def randomize_parameters(self):
        pass

def preprocess(save_path):
    print('preprocessing')
    norm_method = Normalize([114.7748, 107.7354, 99.4750], [1, 1, 1])

    spatial_transform = Compose([
        Scale(int(112 / 1.0)),
        CornerCrop(112, 'c'),
        ToTensor(1), norm_method
    ])
    temporal_transform = TemporalRandomCrop(16, 1)
    target_transform = VideoID()

    test_data = get_test_set(opt, spatial_transform, temporal_transform,
                             target_transform)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=opt.inference_batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              pin_memory=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file = open(opt.info_path, 'w')
    cid = 0
    for i, (inputs, targets) in enumerate(test_loader):
        if(len(targets) == opt.inference_batch_size):
            info = ''
            for j in range(len(targets)):
                info = info + targets[j] + ' '
            batch_bin = inputs.cpu().numpy()
            path_bin = str(save_path) + '/' + str(cid) + '.bin'
            cid = cid + 1
            batch_bin.tofile(path_bin)
            file.write(info)
            file.write('\n')
        if (i % 1000) == 0:
            print('[{}/{}]'.format(i+1, len(test_loader)))
    print('preprocess finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess of 3D-ResNets')
    parser.add_argument('--video_path', default='C:/Users/17270/Efficient-3DCNNs-master-2/annotation_UCF101/UCF-101-image/UCF-101-image', type=Path, help='Directory path of videos')
    parser.add_argument('--annotation_path', default='ucf101_01.json', type=Path, help='Annotation file path')
    parser.add_argument('--dataset', default='ucf101', type=str, help='Used dataset (activitynet | kinetics | ucf101 | hmdb51)')
    parser.add_argument('--test_subset', default='val', type=str, help='Used subset in inference (train | val | test)')
    parser.add_argument('--output_path', default='zjbintt', type=Path, help='Directory path of binary output data')
    parser.add_argument('--info_path', default='zjbin1.info', type=Path, help='Directory path of binary output data')
    parser.add_argument('--inference_batch_size', default=16, type=int, help='Batch Size for inference. 0 means this is the same as batch_size.')
    parser.add_argument('--sample_duration', default=16, type=int, help='Temporal duration of inputs')
    opt = parser.parse_args()

    preprocess(opt.output_path)
