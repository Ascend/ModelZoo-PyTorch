# Copyright 2022 Huawei Technologies Co., Ltd
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
import tqdm
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

import mmcv
import torch.nn.functional as F
from mmseg.ops import resize
from mmseg.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
                        description='merge windows and compute metrics.')
    parser.add_argument('--config', type=str, required=True,
                        help='path to config file.')
    parser.add_argument('--infer-results', type=str, required=True, 
                        help='directory path for saving inference results.')
    args = parser.parse_args()
    return args


def slide_merge(windows, cfg):

    num_classes = cfg.model.decode_head.num_classes
    align_corners = cfg.model.decode_head.align_corners
    crop_size = cfg.model.test_cfg.crop_size

    # Parse file name information
    img_msg_list = Path(windows[0]).name.replace('_0.bin', '').split('-')
    batch_size, h_img, w_img = map(int, img_msg_list[3].split(','))
    ori_shape = list(map(int, img_msg_list[2].split(',')))

    # init the result matrix
    preds = torch.zeros((batch_size, num_classes, h_img, w_img))
    count_mat = torch.zeros((batch_size, 1, h_img, w_img))

    # traverse all the window of a img_bin
    for window_path in windows:
        img_msg_list = Path(window_path).name.replace('_0.bin', '').split('-')
        x1, y1, x2, y2 = map(int, img_msg_list[1].split(','))
        crop_seg_logit = np.fromfile(window_path, dtype=np.float32)
        crop_seg_logit = crop_seg_logit.reshape(1, num_classes, *crop_size)
        crop_seg_logit = torch.Tensor(crop_seg_logit).float()
        preds += F.pad(crop_seg_logit, 
                       (int(x1), int(preds.shape[3] - x2), 
                        int(y1), int(preds.shape[2] - y2)))
        count_mat[:, :, y1:y2, x1:x2] += 1

    assert (count_mat == 0).sum() == 0
    if torch.onnx.is_in_onnx_export():
        count_mat = torch.from_numpy(
            count_mat.cpu().detach().numpy())
    preds = preds / count_mat

    # rescale
    resize_shape = [h_img, w_img, 3]
    preds = preds[:, :, :resize_shape[0], :resize_shape[1]]
    preds = resize(
        preds, 
        size=tuple(ori_shape[:2]), 
        mode='bilinear',
        align_corners=align_corners, 
        warning=False
    )
    output = F.softmax(preds, dim=1)
    seg_pred = output.argmax(dim=1)
    seg_pred = seg_pred.cpu().numpy()
    seg_pred = list(seg_pred)
    return seg_pred


def compute_metrics(config_path, infer_result_dir):

    infer_result_dir = Path(infer_result_dir)
    image2windows = defaultdict(list)
    for file in infer_result_dir.iterdir():
        img_stem = file.stem.split('-', 1)[0]
        image2windows[img_stem].append(file.__str__())

    cfg = mmcv.Config.fromfile(config_path)
    cfg.merge_from_dict({
        'model.test_cfg.mode': 'slide', 
        'model.test_cfg.crop_size': (512, 512), 
        'model.test_cfg.stride': (384, 384)
    })
    dataset = build_dataset(cfg.data.test)

    results = []
    for item in tqdm.tqdm(dataset, desc='Processing'):
        img_stem = Path(item['img_metas'][0].data['ori_filename']).stem
        windows = image2windows[img_stem]
        result = slide_merge(windows, cfg)
        results.extend(result)

    metric = dataset.evaluate(results, metric='mIoU')


if __name__ == '__main__':
    args = parse_args()
    compute_metrics(args.config, args.infer_results)
