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
import mmcv
import numpy as np
from argparse import ArgumentParser
from mmpose.datasets import TopDownCocoDataset
from mmpose.core.evaluation import keypoints_from_heatmaps


def _box2cs(box, image_size):
    x, y, w, h = box[:4]

    aspect_ratio = 1. * image_size[0] / image_size[1]
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / 200.0, h * 1.0 / 200.0], dtype=np.float32)
    scale = scale * 1.25
    return center, scale


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', help='config file path',
                        default='UniFormer/pose_estimation/exp/top_down_256x192_global_base/config.py')
    parser.add_argument('--dataset', help='dataset path',
                        default='data/coco')
    parser.add_argument('--bin', help='result bin file path',
                        default='result')
    args = parser.parse_args()

    dataset_path = args.dataset
    bin_path = args.bin
    latest_result = os.listdir(bin_path)
    latest_result.sort()
    bin_path = os.path.join(bin_path, latest_result[-1])

    cfg = mmcv.Config.fromfile(args.config)
    image_size = cfg.data_cfg['image_size']
    heatmap_size = cfg.data_cfg['heatmap_size']
    ann_file = os.path.join(dataset_path, 'annotations/person_keypoints_val2017.json')
    img_prefix = os.path.join(dataset_path, 'val2017/')

    dataset = TopDownCocoDataset(
        ann_file=ann_file,
        img_prefix=img_prefix,
        data_cfg=cfg.data_cfg,
        pipeline=[],
        test_mode=True)
    coco = dataset.coco

    outputs = []

    # process each image
    for image_id in coco.imgs.keys():
        # get bounding box annotations
        image = coco.loadImgs(image_id)[0]
        image_name = os.path.join(img_prefix, image['file_name'])
        ann_ids = coco.getAnnIds(image_id, iscrowd=False)

        output = {'preds': [], 'boxes': [], 'image_paths': [],
                  'output_heatmap': None, 'bbox_ids': []}

        for ann_id in ann_ids:
            ann = coco.anns[ann_id]
            # no keypoints, skip
            if 'keypoints' not in ann:
                continue
            if max(ann['keypoints']) == 0:
                continue
            if 'num_keypoints' in ann and ann['num_keypoints'] == 0:
                continue

            # bbox format is 'xywh'
            bbox = ann['bbox']
            center, scale = _box2cs(bbox, image_size)

            heatmap = np.fromfile(os.path.join(
                bin_path, f'{ann_id}_output_0.bin'), dtype=np.float32)
            heatmap = np.reshape(
                heatmap, [1, 17, heatmap_size[1], heatmap_size[0]])

            preds, maxvals = keypoints_from_heatmaps(
                heatmap, [center], [scale])
            keypoints = np.concatenate((preds, maxvals), axis=2)

            output['preds'].append(keypoints[0])
            output['boxes'].append(np.array(
                [center[0], center[1], scale[0], scale[1], scale[0] * scale[1] * 200 * 200, 1.0]))
            output['image_paths'].append(image_name)
            output['bbox_ids'].append(len(output['bbox_ids']))

        outputs.append(output)

    os.makedirs('tmp', exist_ok=True)
    dataset.evaluate(outputs, 'tmp', 'mAP')


if __name__ == '__main__':
    main()
