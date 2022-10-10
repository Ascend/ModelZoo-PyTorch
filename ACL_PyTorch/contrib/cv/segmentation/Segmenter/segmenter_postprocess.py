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


import sys
from pathlib import Path
import tqdm

import yaml
import numpy as np
import torch

import mmcv

from segm.model.utils import merge_windows
from segm.metrics import compute_metrics
from segm.data.utils import IGNORE_LABEL


def get_predict(output_dir):

    B = 6
    C = 19
    window_size = 768
    ori_shape = torch.Size((1024, 2048))
    windows = {
        'anchors': [(0, 0), (0, 736), (0, 1280), (256, 0), (256, 736), (256, 1280)],
        'flip': torch.tensor([False]), 'shape': (1024, 2048)
    }

    seg_pred_maps = {}
    for res_file in tqdm.tqdm(Path(output_dir).iterdir(), desc="Reading prediction"):
        if res_file.suffix != '.bin':
            continue
        pred_seg = np.fromfile(res_file, dtype = np.float32)
        pred_seg = torch.from_numpy(pred_seg.reshape(C, window_size, window_size))
        key, w_idx = res_file.name.replace('_0.bin', '').rsplit('_', 1)
        w_idx = int(w_idx)
        if key not in seg_pred_maps:
            seg_pred_maps[key] = torch.zeros((B, C, window_size, window_size))
        seg_pred_maps[key][w_idx:w_idx+1] = pred_seg

    for key, pred_seg in tqdm.tqdm(seg_pred_maps.items(), desc="Processing prediction"):
        windows["seg_maps"] = pred_seg
        im_seg_map = merge_windows(windows, window_size, ori_shape)
        seg_pred_maps[key] = im_seg_map.argmax(0)

    return seg_pred_maps


def get_groudtruth(gt_file):

    ignore_label = 255
    gt_seg_maps = {}
    for line in open(gt_file, "r"):
        src_path, seg_map = line.strip().split("\t")
        gt_seg_map = mmcv.imread(seg_map, flag="unchanged", backend="pillow")
        gt_seg_map[gt_seg_map == ignore_label] = IGNORE_LABEL
        key = Path(src_path).stem
        gt_seg_maps[key] = gt_seg_map

    return gt_seg_maps


def save_metrics(scores, metrics_file):
    scores["inference"] = "single_scale"
    suffix = "ss"
    scores["cat_iou"] = np.round(100 * scores["cat_iou"], 2).tolist()
    for k, v in scores.items():
        if k != "cat_iou" and k != "inference":
            scores[k] = v.item()
        if k != "cat_iou":
            print(f"{k}: {scores[k]}")
    scores_str = yaml.dump(scores)
    with open(metrics_file, "w") as f:
        f.write(scores_str)


def main():

    import argparse
    parser = argparse.ArgumentParser('compute metrics.')
    parser.add_argument('--result-dir', type=str, required=True,
        help='path to infer results.')
    parser.add_argument('--gt-path', type=str, required=True,
        help='path to groudtruth.')
    parser.add_argument('--metrics-path', type=str, default=None,
        help='a text file, to save metrics.')
    args = parser.parse_args()

    seg_pred_maps = get_predict(args.result_dir)
    seg_gt_maps = get_groudtruth(args.gt_path)
    scores = compute_metrics(
        seg_pred_maps,
        seg_gt_maps,
        n_cls =19,
        ignore_index=IGNORE_LABEL,
        ret_cat_iou=True,
        distributed=False,
    )

    if args.metrics_path:
        save_metrics(scores, args.metrics_path)


if __name__ == '__main__':
    main()

