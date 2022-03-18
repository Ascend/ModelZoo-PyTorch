# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import json
import argparse
from tqdm import tqdm
import numpy as np
import torch

sys.path.append('./AlphaPose')
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.metrics import evaluate_mAP
from alphapose.utils.transforms import get_func_heatmap_to_coord, flip_heatmap
from alphapose.utils.pPose_nms import oks_pose_nms


parser = argparse.ArgumentParser(description='AlphaPose Postprocess')
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    default='./AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml',
                    type=str)
parser.add_argument('--dataroot', dest='dataroot',
                    help='data root dirname', default='./data/coco',
                    type=str)
parser.add_argument('--dump_dir', dest='dump_dir',
                    help='dump dir for bin files',
                    required=True,
                    type=str)
parser.add_argument('--dump_dir_flip', dest='dump_dir_flip',
                    help='dump dir for fliped bin files',
                    required=True,
                    type=str)
parser.add_argument('--gpus',
                    help='gpus',
                    default='-1',
                    type=str)
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")

opt = parser.parse_args()

gpus = [int(i) for i in opt.gpus.split(',')]
opt.gpus = [gpus[0]]
opt.device = torch.device("cuda:" + str(opt.gpus[0]) if opt.gpus[0] >= 0 else "cpu")


def get_output_data(dump_dir, idx, dtype=np.float32):
    output_shape = [1, 17, 64, 48]
    input_file = os.path.join(dump_dir, "{:0>12d}_1.bin".format(idx))
    input_data = np.fromfile(input_file, dtype=dtype).reshape(output_shape)
    return input_data


def postprocess(cfg, heatmap_to_coord, batch_size):
    det_dataset = builder.build_dataset(
        cfg.DATASET.TEST, preset_cfg=cfg.DATA_PRESET, train=False, opt=opt)
    eval_joints = det_dataset.EVAL_JOINTS

    det_loader = torch.utils.data.DataLoader(
        det_dataset, batch_size=batch_size, shuffle=False, num_workers=20, drop_last=False)
    kpt_json = []

    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE

    for idx, det_data in tqdm(enumerate(det_loader), dynamic_ncols=True):
        crop_bboxes = det_data[1]
        bboxes = det_data[2]
        img_ids = det_data[3]
        scores = det_data[4]
        output = get_output_data(opt.dump_dir, idx)

        # flip_test
        output_flip = get_output_data(opt.dump_dir_flip, idx)
        output_flip = flip_heatmap(torch.tensor(output_flip), det_dataset.joint_pairs, shift=True).numpy()
        pred_flip = output_flip[:, eval_joints, :, :]

        pred = output
        assert len(pred.shape) == 4
        pred = pred[:, eval_joints, :, :]

        for i in range(output.shape[0]):
            bbox = crop_bboxes[i].tolist()
            pose_coords, pose_scores = heatmap_to_coord(
                pred[i], bbox, hms_flip=pred_flip[i], hm_shape=hm_size, norm_type=norm_type)

            keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
            keypoints = keypoints.reshape(-1).tolist()

            data = dict()
            data['bbox'] = bboxes[i, 0].tolist()
            data['image_id'] = int(img_ids[i])
            data['area'] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            data['score'] = float(scores[i])
            data['category_id'] = 1
            data['keypoints'] = keypoints

            kpt_json.append(data)

    kpt_json = oks_pose_nms(kpt_json)

    with open('./exp/json/validate_rcnn_kpt.json', 'w') as fid:
        json.dump(kpt_json, fid)
    res = evaluate_mAP('./exp/json/validate_rcnn_kpt.json', ann_type='keypoints', ann_file=os.path.join(cfg.DATASET.VAL.ROOT, cfg.DATASET.VAL.ANN))
    return res


if __name__ == '__main__':
    config = update_config(opt.cfg)
    heatmap = get_func_heatmap_to_coord(config)
    detbox_AP = postprocess(config, heatmap, 1)
    print('det box: {} mAP #####'.format(detbox_AP))
