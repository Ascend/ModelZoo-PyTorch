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
import cv2
import mmcv
import numpy as np
from argparse import ArgumentParser
from xtcocotools.coco import COCO
from mmpose.core.post_processing import get_affine_transform
from mmpose.apis.inference import _box2cs
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', help='config file path',
                        default='UniFormer/pose_estimation/exp/top_down_256x192_global_base/config.py')
    parser.add_argument('--dataset', help='dataset path',
                        default='data/coco')
    parser.add_argument('--bin', help='preprocessed bin file path',
                        default='val2017_bin')
    args = parser.parse_args()

    dataset_path = args.dataset
    bin_path = args.bin
    os.makedirs(bin_path, exist_ok=True)

    cfg = mmcv.Config.fromfile(args.config)
    image_size = cfg.data_cfg['image_size']
    ann_file = os.path.join(dataset_path, 'annotations/person_keypoints_val2017.json')
    img_prefix = os.path.join(dataset_path, 'val2017')

    coco = COCO(ann_file)

    # process each image
    for image_id in tqdm(coco.imgs.keys()):
        # get bounding box annotations
        image = coco.loadImgs(image_id)[0]
        # print(image)
        image_name = os.path.join(img_prefix, image['file_name'])
        ori_img = mmcv.imread(image_name)
        ann_ids = coco.getAnnIds(image_id, iscrowd=False)

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
            center, scale = _box2cs(cfg, bbox)

            img = ori_img.copy()
            # TopDownAffine
            trans = get_affine_transform(center, scale, 0, image_size)
            img = cv2.warpAffine(
                img,
                trans, (int(image_size[0]), int(image_size[1])),
                flags=cv2.INTER_LINEAR)

            # NormalizeTensor
            mean = np.array([0.485, 0.456, 0.406]) * 255
            std = np.array([0.229, 0.224, 0.225]) * 255
            img = mmcv.imnormalize(img, mean, std)

            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, 0)

            img.tofile(os.path.join(bin_path, f'{ann_id}.bin'))


if __name__ == '__main__':
    main()
