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
"""Validation script."""

import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm

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


def validate(m, heatmap_to_coord, batch_size=20):
    det_dataset = builder.build_dataset(cfg.DATASET.TEST, preset_cfg=cfg.DATA_PRESET, train=False, opt=opt)
    eval_joints = det_dataset.EVAL_JOINTS

    det_loader = torch.utils.data.DataLoader(
        det_dataset, batch_size=batch_size, shuffle=False, num_workers=20, drop_last=False)
    kpt_json = []
    m.eval()

    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE

    for inps, crop_bboxes, bboxes, img_ids, scores, imghts, imgwds in tqdm(det_loader, dynamic_ncols=True):
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
        #     output_flip = flip_heatmap(m(inps_flip), det_dataset.joint_pairs, shift=True)
        #     pred_flip = output_flip[:, eval_joints, :, :]
        # else:
        #     output_flip = None

        pred = output
        assert pred.dim() == 4
        pred = pred[:, eval_joints, :, :]

        for i in range(output.shape[0]):
            bbox = crop_bboxes[i].tolist()
            pose_coords, pose_scores = heatmap_to_coord(
                pred[i], bbox,  hm_shape=hm_size, norm_type=norm_type)

            keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
            keypoints = keypoints.reshape(-1).tolist()

            data = dict()
            data['bbox'] = bboxes[i, 0].tolist()
            data['image_id'] = int(img_ids[i])
            data['area'] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            # data['score'] = float(scores[i] + np.mean(pose_scores) + np.max(pose_scores))
            data['score'] = float(scores[i])
            data['category_id'] = 1
            data['keypoints'] = keypoints

            kpt_json.append(data)

    # kpt_json = oks_pose_nms(kpt_json)

    with open('./exp/validate_rcnn_kpt.json', 'w') as fid:
        json.dump(kpt_json, fid)
    res = evaluate_mAP('./exp/validate_rcnn_kpt.json', ann_type='keypoints', ann_file=os.path.join(cfg.DATASET.VAL.ROOT, cfg.DATASET.VAL.ANN))
    # return res['AP']
    return res



def validate_gt(m, cfg, heatmap_to_coord, batch_size=20):
    gt_val_dataset = builder.build_dataset(cfg.DATASET.VAL, preset_cfg=cfg.DATA_PRESET, train=False)
    eval_joints = gt_val_dataset.EVAL_JOINTS

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=batch_size, shuffle=False, num_workers=20, drop_last=False)
    kpt_json = []
    m.eval()

    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE

    for inps, labels, label_masks, img_ids, bboxes in tqdm(gt_val_loader, dynamic_ncols=True):
        # print(type(inps))
        # torch.save(inps,'./input.pt')
        # print('save success')
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
            # exit()
            kpt_json.append(data)
        # exit()

    with open('./exp/validate_gt_kpt.json', 'w') as fid:
        json.dump(kpt_json, fid)
        # print("write json file success!")
    res = evaluate_mAP('./exp/validate_gt_kpt.json', ann_type='keypoints', ann_file=os.path.join(cfg.DATASET.VAL.ROOT, cfg.DATASET.VAL.ANN))
    # print(res)
    #return res['AP']
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
        with open('exp/exp_test-256x192_res50_lr1e-3_1x.yaml/eval.log','w') as f:
            f.write('##### gt box: {} mAP #####'.format(gt_AP))
        print('##### gt box: {} mAP #####'.format(gt_AP))
        # print('##### det box: {} mAP #####'.format(detbox_AP))

