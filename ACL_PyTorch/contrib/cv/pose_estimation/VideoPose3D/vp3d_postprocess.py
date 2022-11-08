# Copyright 2022 Huawei Technologies Co., Ltd
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

sys.path.append('./VideoPose3D')

from common.loss import mpjpe
import numpy as np
import argparse
import torch
import os.path as osp
import json
import glob


def evaluate(plist, glist, ddic):
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]
    epoch_loss_3d_pos = 0
    N = 0
    for pred_path, gt_path in zip(plist, glist):
        idx = int(gt_path.split('/')[-1].split('_')[0])
        ref_idx = int(pred_path.split('/')[-1].split('_')[0])
        assert idx == ref_idx

        pred = np.fromfile(pred_path, dtype='float32')
        gt = np.fromfile(gt_path, dtype='float32')
        pred = pred.reshape(2, -1, 17, 3)
        if ddic[idx] > 0:
            pred = pred[:, :-ddic[idx]]
        gt = gt.reshape(2, -1, 17, 3)
        assert pred.shape == gt.shape, f"pred.shape:{pred.shape}, gt.shape:{gt.shape}"

        pred = torch.from_numpy(pred)
        gt = torch.from_numpy(gt)
        # augmentation
        pred[1, :, :, 0] *= -1
        pred[1, :, joints_left + joints_right] = pred[1, :, joints_right + joints_left]
        pred = torch.mean(pred, dim=0, keepdim=True)

        gt[:, :, 0] = 0
        gt = gt[:1]
        error = mpjpe(pred, gt)
        epoch_loss_3d_pos += gt.shape[0] * gt.shape[1] * error.item()
        N += gt.shape[0] * gt.shape[1]
    return (epoch_loss_3d_pos / N) * 1000


def get_all_data(pre_data_dir, infer_result_dir):
    pred_file = osp.join(infer_result_dir, '*.bin')
    gt_file = osp.join(pre_data_dir, 'ground_truths/*.bin')
    delta_file = osp.join(pre_data_dir, 'delta_dict_padding.json')
    

    print("Loading relavent data...")

    with open(delta_file, 'r') as f:
        delta_dict = json.load(f)
    predlist = glob.glob(pred_file)
    gtlist = glob.glob(gt_file)

    pred_dict = {}
    for pred in predlist:
        act = pred.split('_')[-2]
        if not act in pred_dict:
            pred_dict[act] = []
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
        ddict[idx] = v

    return pred_dict, gt_dict, ddict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='postprocess.')
    parser.add_argument('--preprocess-data', type=str, required=True, metavar='PATH', 
                         help='path to preprocessed dataset.')
    parser.add_argument('--infer-results', type=str, required=True, metavar='PATH', 
                         help='path to results of inference.')
    args = parser.parse_args()

    pred_dict, gt_dict, ddict = get_all_data(args.preprocess_data, args.infer_results)
    actions = list(gt_dict.keys())
    # evaluate by action
    print("Start inference...")
    elist = []
    for act in actions:
        plist = pred_dict[act]
        glist = gt_dict[act]
        plist = sorted(plist)
        glist = sorted(glist)
        e = evaluate(plist, glist, ddict)
        elist.append(e)

    print('==== Validation Results ====')
    print(f"Protocol #1 (MPJPE) action-wise average:{round(np.mean(elist), 1)}mm")
