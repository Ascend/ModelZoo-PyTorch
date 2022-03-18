# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------
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
# ===========================================================================
"""Validation script."""
import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm
import cv2

from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.metrics import evaluate_mAP
from alphapose.utils.transforms import (flip, flip_heatmap,
                                        get_func_heatmap_to_coord)
from alphapose.utils.pPose_nms import oks_pose_nms

import torch.npu
CALCULATE_DEVICE = "npu:0"
torch.npu.set_device(CALCULATE_DEVICE)

# CALCULATE_DEVICE = "cpu"

parser = argparse.ArgumentParser(description='AlphaPose Validate')
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)
parser.add_argument('--checkpoint',
                    help='checkpoint file name',
                    required=True,
                    type=str)
parser.add_argument('--gpus',
                    help='gpus',
                    type=str)
parser.add_argument('--batch',
                    help='validation batch size',
                    type=int)
parser.add_argument('--flip-test',
                    default=False,
                    dest='flip_test',
                    help='flip test',
                    action='store_true')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")

opt = parser.parse_args()
cfg = update_config(opt.cfg)

# gpus = [int(i) for i in opt.gpus.split(',')]
# opt.gpus = [gpus[0]]
# opt.device = torch.device("cuda:" + str(opt.gpus[0]) if opt.gpus[0] >= 0 else "cpu")



def validate_gt(m, cfg, heatmap_to_coord, batch_size=20):
    gt_val_dataset = builder.build_dataset(cfg.DATASET.DEMO, preset_cfg=cfg.DATA_PRESET, train=False)
    eval_joints = gt_val_dataset.EVAL_JOINTS

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=1, shuffle=False, num_workers=20, drop_last=False)
    kpt_json = []
    m.eval()

    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE

    for inps, labels, label_masks, img_ids, bboxes,img_path in gt_val_loader:
        if isinstance(inps, list):
            inps = [inp.to(CALCULATE_DEVICE) for inp in inps]
        else:
            inps = inps.to(CALCULATE_DEVICE)
        output = m(inps)
        # if opt.flip_test:
        #     if isinstance(inps, list):
        #         inps_flip = [flip(inp).to(CALCULATE_DEVICE) for inp in inps]
        #     else:
        #         inps_flip = flip(inps).to(CALCULATE_DEVICE)
        #     output_flip = flip_heatmap(m(inps_flip), gt_val_dataset.joint_pairs, shift=True)
        #     pred_flip = output_flip[:, eval_joints, :, :]
        # else:
        #     output_flip = None

        pred = output.float()
        # pred = output

        assert pred.dim() == 4
        pred = pred[:, eval_joints, :, :]

        # print('size different:')
        # print(pred[0].size())
        # print(pred[0][gt_val_dataset.EVAL_JOINTS].size())


        for i in range(output.shape[0]):
            bbox = bboxes[i].tolist()
            pose_coords, pose_scores = heatmap_to_coord(
                pred[i], bbox, hm_shape=hm_size, norm_type=norm_type)
            # print(pose_coords)
            # print(pose_scores)

            keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
            keypoints = keypoints.reshape(-1).tolist()

            data = dict()
            data['bbox'] = bboxes[i].tolist()
            data['image_id'] = int(img_ids[i])
            data['score'] = float(np.mean(pose_scores) + np.max(pose_scores))
            data['category_id'] = 1
            data['keypoints'] = keypoints
            img = cv2.imread(img_path[0])
            for i in range(17):
                x=int(keypoints[i*3+0])
                y=int(keypoints[i*3+1])
                score=keypoints[i*3+2]
                if x>0 and y>0 and score>0.3:
                    pos=(x,y)
                    cv2.circle(img, pos, 5, color=(0, 255, 0))
            img_name = 'out.jpg'
            cv2.imwrite(img_name, img)
            return 
            # exit()
        # exit()

    return res



if __name__ == "__main__":
    m = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    m = m.to(CALCULATE_DEVICE)

    print(f'Loading model from {opt.checkpoint}...')
    m.load_state_dict(torch.load(opt.checkpoint,map_location='npu:0'))
    # m.load_state_dict(torch.load(opt.checkpoint))
    # m = torch.nn.DataParallel(m, device_ids=gpus).to(CALCULATE_DEVICE)

    heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    with torch.no_grad():
        gt_AP = validate_gt(m, cfg, heatmap_to_coord, opt.batch)
