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

import sys
sys.path.append('./VideoPose3D')

from common.loss import mpjpe
import numpy as np
import argparse
import torch
import os.path as osp
import json
import glob


def evaluate(plist,glist,ddic):
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]
    epoch_loss_3d_pos = 0
    N = 0
    for pred_path, gt_path in zip(plist,glist):
#         print(gt_path)
        idx = int(gt_path.split('/')[-1].split('_')[0])
        ref_idx = int(pred_path.split('/')[-1].split('_')[0])
        assert idx == ref_idx
        
        pred = np.fromfile(pred_path, dtype='float32')
        gt = np.fromfile(gt_path, dtype='float32')
        pred = pred.reshape(2,-1,17,3)
        if ddic[idx] > 0:
            pred = pred[:,:-ddic[idx]]
        gt = gt.reshape(2,-1,17,3)
        assert pred.shape == gt.shape, f"pred.shape:{pred.shape},gt.shape:{gt.shape},pred:{pred_path},gt:{gt_path},delta:{ddic[idx]}"

        pred = torch.from_numpy(pred)
        gt = torch.from_numpy(gt)
        # augmentation
        pred[1,:,:,0] *= -1
        pred[1, :, joints_left + joints_right] = pred[1, :, joints_right + joints_left]
        pred = torch.mean(pred, dim=0, keepdim=True)

        gt[:,:,0] = 0
        gt = gt[:1]
        error = mpjpe(pred,gt)
        epoch_loss_3d_pos += gt.shape[0]*gt.shape[1] * error.item()
        N += gt.shape[0] * gt.shape[1]
    return (epoch_loss_3d_pos / N)*1000


def get_all_data(args):
    data_root = args.dataset
    delta_file = 'delta_dict_padding.json'
    pred_file = f'outputs/{args.out}/*.bin'
    gt_file = 'ground_truths/*.bin'

    print("Loading relavent data...")

    with open(osp.join(data_root, delta_file),'r') as f:
        delta_dict = json.load(f)
    predlist = glob.glob(osp.join(data_root, pred_file))
    gtlist = glob.glob(osp.join(data_root, gt_file))
    
    pred_dict = {}
    for pred in predlist:
        act = pred.split('_')[-3]
        if not act in pred_dict:
            pred_dict[act]=[]
        pred_dict[act].append(pred)

    gt_dict = {}
    for gt in gtlist:
        act = gt.split('_')[-1].split('.')[0]
        if not act in gt_dict:
            gt_dict[act] = []
        gt_dict[act].append(gt)

    ddict = {}
    for k, v in delta_dict.items():
        idx = int(k.split("_")[0])
        ddict[idx]=v

    return pred_dict, gt_dict, ddict

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Preprocessing of vp3d dataset')
    parser.add_argument('-d', '--dataset', default='./preprocessed_data',
                        type=str, metavar='PATH', help='path to preprocessed dataset')
    parser.add_argument('-o', '--out', default='./preprocessed_data/outputs')
    args = parser.parse_args()
    pred_dict, gt_dict, ddict = get_all_data(args)

    actions = list(gt_dict.keys())
    # evaluate by action
    print("Start inference...")
    elist = []
    for act in actions:
        plist = pred_dict[act]
        glist = gt_dict[act]
        plist = sorted(plist)
        glist = sorted(glist)
        e = evaluate(plist,glist,ddict)
        elist.append(e)

    print('==== Validation Results ====')
    print(f"Protocol #1   (MPJPE) action-wise average:{round(np.mean(elist),1)}mm")