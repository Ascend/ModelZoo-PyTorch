# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================

from genericpath import exists
from itertools import zip_longest
import sys
sys.path.append('./VideoPose3D')

import os
import os.path as osp
import numpy as np
import argparse
import json
from common.h36m_dataset import Human36mDataset
from common.utils import fetch_actions
from common.generators import UnchunkedGenerator
from common.camera import prepare_data


def preprocess(args):
    path_3d = osp.join(args.dataset,'data_3d_h36m.npz')
    path_2d = osp.join(args.dataset,'data_2d_h36m_cpn_ft_h36m_dbb.npz')

    print('Loading dataset...')
    dataset = Human36mDataset(path_3d)
    keypoints = np.load(path_2d, allow_pickle=True)

    # proccessed data save path
    input_path = osp.join(args.save, 'inputs')
    gt_path = osp.join(args.save, "ground_truths")
    if not osp.exists(input_path):
        os.makedirs(input_path)
    if not osp.exists(gt_path):
        os.makedirs(gt_path)

    dataset, keypoints, kps_left, kps_right, joints_left, joints_right = prepare_data(dataset, keypoints)

    actions = {}
    subjects = args.subjects_test.split(',')
    for sub in subjects:
        for act in dataset[sub].keys():
            act_name = act.split(' ')[0]
            if act_name not in actions:
                actions[act_name] = []
            actions[act_name].append((sub, act))
    idx_count = 0
    delta_dict = {}
    print('Saving processed data...')
    for act_key in actions.keys():
        poses_act, poses_2d_act = fetch_actions(args, actions[act_key], keypoints, dataset)

        test_generator = UnchunkedGenerator(None, poses_act, poses_2d_act,
                                     pad=121, causal_shift=0, augment=True,
                                     kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

        
        for _, batch, batch_2d, delta in test_generator.next_epoch():
            idx_count += 1
            save_gt = osp.join(gt_path, "{:03d}_{}.bin".format(idx_count, act_key))
            save_input = osp.join(input_path, "{:03d}_{}.bin".format(idx_count, act_key))
            batch.tofile(save_gt)
            batch_2d.tofile(save_input)

            delta_dict["{:03d}_{}".format(idx_count, act_key)] = delta
    
    delta_path = osp.join(args.save, 'delta_dict_padding.json')
    with open(delta_path, 'w') as out:
        json.dump(delta_dict, out)

    print("All data saved.")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing of vp3d dataset')
    parser.add_argument('-d', '--dataset', default='./VideoPose3D/data',
                        type=str, metavar='PATH', help='path to dataset')
    parser.add_argument('-ste', '--subjects-test', default='S9,S11', type=str, metavar='LIST', 
                        help='test subjects separated by comma')
    parser.add_argument('-s', '--save', default='./preprocessed_data')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', 
                        help='downsample frame rate by factor (semi-supervised)')
    args = parser.parse_args()
    preprocess(args)
