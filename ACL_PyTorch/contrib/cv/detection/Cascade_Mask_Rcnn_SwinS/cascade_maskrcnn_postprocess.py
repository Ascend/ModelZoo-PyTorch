# Copyright 2021 Huawei Technologies Co., Ltd
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


import argparse
import json

import numpy as np
import torch
from tqdm import tqdm
from mmdet.core import bbox2result, encode_mask_results
from mmdet.datasets import CocoDataset
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import _do_paste_mask
import os


def get_map():
    data_dict = {}
    path = args.bin_file_path+'/sumary.json'
    with open(path, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
        data = json_data.get('filesinfo')  # 5000个dict
        for key in data.keys():
            data_dict2 = data.get(key)
            infile = data_dict2.get('infiles')
            name = infile[0].split('/')[2]
            print('name: ', name)
            outfiles = data_dict2.get('outfiles')
            data_dict[name] = outfiles
            print('outfiles: ', outfiles)
    return data_dict


def postprecess():
    data_dict = get_map()
    dataset = CocoDataset(ann_file=args.ann_file_path, pipeline=[])
    bin_path = args.bin_file_path
    latest_result = os.listdir(bin_path)
    latest_result.sort()
    model_h = args.input_height
    model_w = args.input_width

    results = []
    for data_info in tqdm(dataset.data_infos):
        file_name = data_info['file_name']
        ori_h = data_info['height']
        ori_w = data_info['width']
        scalar_ratio = min(model_h / ori_h, model_w / ori_w)

        path_base = file_name.split('.')[0]+'.bin'
        print('path_base: ', path_base)
        path_item_bboxes = data_dict.get(path_base)[0]
        bboxes = np.fromfile(path_item_bboxes, dtype=np.float32)
        bboxes = np.reshape(bboxes, [100, 5])
        path_item_labels = data_dict.get(path_base)[1]
        labels = np.fromfile(path_item_labels, dtype=np.int64)
        path_item_mask = data_dict.get(path_base)[2]
        mask_pred = np.fromfile(path_item_mask, dtype=np.float32)
        mask_pred = np.reshape(mask_pred, [100, 1, 28, 28])

        bboxes[..., 0:4] = bboxes[..., 0:4] / scalar_ratio

        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        mask_pred = torch.from_numpy(mask_pred).to(device)

        bboxes_tensor = torch.from_numpy(bboxes[..., 0:4]).to(device)
        img_h = ori_h
        img_w = ori_w
        N = 100
        threshold = 0.5

        masks, spatial_inds = _do_paste_mask(
            mask_pred,
            bboxes_tensor,
            img_h,
            img_w,
            skip_empty=device.type == 'cpu')
        masks = (masks >= threshold).to(dtype=torch.bool)

        im_mask = torch.zeros(N, img_h, img_w, device=device, dtype=torch.bool)
        im_mask[(torch.arange(N),) + spatial_inds] = masks
        segms = im_mask.squeeze(1).cpu().numpy()

        cls_segms = [[] for _ in range(80)]
        for label, segm in zip(labels, segms):
            cls_segms[label].append(segm)
        cls_bboxes = bbox2result(bboxes, labels, 80)
        results.append((cls_bboxes, encode_mask_results(cls_segms)))

    dataset.evaluate(results, metric=[
                     'bbox', 'segm'], classwise=True, jsonfile_prefix='./')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann_file_path", default="./data/coco/annotations/instances_val2017.json")
    parser.add_argument("--bin_file_path", default="./result_ais/")
    parser.add_argument("--input_height", default=800, type=int)
    parser.add_argument("--input_width", default=1344, type=int)
    args = parser.parse_args()
    postprecess()
