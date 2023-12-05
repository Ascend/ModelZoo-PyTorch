# Copyright 2023 Huawei Technologies Co., Ltd
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

import mmcv
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import pycocotools.mask as mask_util
from mmdet.core import coco_eval, results2json, results2json_segm
from mmdet.datasets import build_dataset
import os.path as osp
from tqdm import tqdm


ann_file = 'annotations/instances_val2017.json'
img_prefix = 'val2017/'


def get_masks(result, num_classes=80):
    for cur_result in result:
        masks = [[] for _ in range(num_classes)]
        if cur_result is None:
            return masks
        seg_pred = cur_result[0].astype(np.uint8)
        cate_label = cur_result[1].astype(np.int)
        cate_score = cur_result[2].astype(np.float)
        num_ins = seg_pred.shape[0]
        for idx in range(num_ins):
            cur_mask = seg_pred[idx, ...]
            rle = mask_util.encode(
                np.array(cur_mask[:, :, np.newaxis], order='F'))[0]
            rst = (rle, cate_score[idx])
            masks[cate_label[idx]].append(rst)
        return masks


def handle_seg(seg, img_shape, ori_shape, input_shape=(800, 1216), mask_thr=0.5):
    seg = torch.tensor(seg)
    h, w, = img_shape
    pad_left = (input_shape[1] - w) // 2
    pad_top = (input_shape[0] - h) // 2
    seg = F.interpolate(seg.unsqueeze(0),
                        size=input_shape,
                        mode='bilinear')[:, :, pad_top:pad_top + h, pad_left:pad_left + w]

    seg = F.interpolate(seg,
                        size=ori_shape[:2],
                        mode='bilinear').squeeze(0)
    seg = seg > mask_thr
    return seg.numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path')
    parser.add_argument('--model_config')
    parser.add_argument('--bin_data_path')
    parser.add_argument('--meta_info')
    parser.add_argument('--net_out_num', type=int)
    parser.add_argument("--model_input_height", type=int,
                        help='input tensor height')
    parser.add_argument("--model_input_width", type=int,
                        help='input tensor width')

    args = parser.parse_args()

    cfg = mmcv.Config.fromfile(args.model_config)
    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = args.dataset_path + ann_file
    cfg.data.test.img_prefix = args.dataset_path + img_prefix
    dataset = build_dataset(cfg.data.test)
    num_classes = len(dataset.CLASSES)

    results = []

    with open(args.meta_info, "r") as fp:
        for line in tqdm(fp):
            _, file_path, img_w, img_h, ori_w, ori_h = line.split()
            img_w = int(img_w)
            img_h = int(img_h)
            ori_w = int(ori_w)
            ori_h = int(ori_h)
            file_name = file_path.split("/")[1].replace(".bin", "")
            file_name = osp.join(args.bin_data_path, file_name)
            result = []
            for idx in range(args.net_out_num):
                if idx == 1:
                    result.append(np.fromfile(
                        f"{file_name}_{idx}.bin", dtype=np.int32))
                else:
                    result.append(np.fromfile(
                        f"{file_name}_{idx}.bin", dtype=np.float32))
            result[0].shape = (100, args.model_input_height //
                               4, args.model_input_width // 4)
            result[0] = handle_seg(result[0], (img_h, img_w), (ori_h, ori_w),
                                   (args.model_input_height, args.model_input_width))
            result = get_masks([result], num_classes)
            results.append(result)

    result_files = results2json_segm(dataset, results, "results_solo.pkl")
    coco_eval(result_files, ["segm"], dataset.coco)
