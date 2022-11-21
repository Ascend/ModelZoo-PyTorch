
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

# Copyright (c) Open-MMLab. All rights reserved.    
import argparse
import os.path as osp

import mmcv


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate COCO test image information '
        'for COCO panoptic segmentation.')
    parser.add_argument('data_root', help='Path to COCO annotation directory.')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    data_root = args.data_root
    val_info = mmcv.load(osp.join(data_root, 'panoptic_val2017.json'))
    test_old_info = mmcv.load(
        osp.join(data_root, 'image_info_test-dev2017.json'))

    # replace categories from image_info_test-dev2017.json
    # with categories from panoptic_val2017.json which
    # has attribute `isthing`.
    test_info = test_old_info
    test_info.update({'categories': val_info['categories']})
    mmcv.dump(test_info,
              osp.join(data_root, 'panoptic_image_info_test-dev2017.json'))


if __name__ == '__main__':
    main()
