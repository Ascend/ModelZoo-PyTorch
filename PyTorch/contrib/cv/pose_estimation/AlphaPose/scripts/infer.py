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
from PIL import Image
from torchvision import transforms

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
parser.add_argument('--image', dest='inputimg',
                    help='image-name', 
                    required=True,
                    default="")
parser.add_argument('--gpus',
                    help='gpus',
                    type=str)
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")

opt = parser.parse_args()
cfg = update_config(opt.cfg)

# gpus = [int(i) for i in opt.gpus.split(',')]
# opt.gpus = [gpus[0]]
# opt.device = torch.device("cuda:" + str(opt.gpus[0]) if opt.gpus[0] >= 0 else "cpu")

def inference(m, cfg, heatmap_to_coord, image_tensor, batch_size):
    det_dataset = builder.build_dataset(cfg.DATASET.TEST, preset_cfg=cfg.DATA_PRESET, train=False, opt=opt)
    eval_joints = det_dataset.EVAL_JOINTS

    det_loader = torch.utils.data.DataLoader(
        det_dataset, batch_size=batch_size, shuffle=False, num_workers=20, drop_last=False)
    kpt_json = []
    m.eval()

    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE

    output = m(image_tensor)
    pred = output.float()

    assert pred.dim() == 4
    pred = pred[:, eval_joints, :, :]
    for inps, crop_bboxes, bboxes, img_ids, scores, imghts, imgwds in det_loader:
        for i in range(output.shape[0]):
            bbox = bboxes[i][0].tolist()
            pose_coords, pose_scores = heatmap_to_coord(
                pred[i], bbox, hm_shape=hm_size, norm_type=norm_type)

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
        with open('./result.json', 'w') as fid:
            json.dump(kpt_json, fid)
            print("write json file success!")
            return True
        return False

if __name__ == "__main__":
    m = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print(f'Loading model from {opt.checkpoint}...')
    m.load_state_dict(torch.load(opt.checkpoint,map_location='cpu'))

    m = m.to(CALCULATE_DEVICE)

    heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    preprocess_transform = transforms.Compose([
        transforms.Resize((256,192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    img = Image.open(opt.inputimg).convert('RGB')
    image_tensor = preprocess_transform(img)
    image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.to(CALCULATE_DEVICE)

    with torch.no_grad():
        gt_AP = inference(m, cfg, heatmap_to_coord, image_tensor, 20)
