# Copyright 2020 Huawei Technologies Co., Ltd
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
import argparse
import numpy as np
from tqdm import tqdm
import sys
import datasets

sys.path.append('HRNet-Semantic-Segmentation/lib/')
from config import config
from config import update_config


def parse_args():
    parser = argparse.ArgumentParser(description='HRNet preprocess process.')
    
    parser.add_argument('--cfg',
                        default='HRNet-Semantic-Segmentation/experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml',
                        help='experiment configure file name',
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--src_path')
    parser.add_argument('--save_path')
                        
    args = parser.parse_args()
    update_config(config, args)

    return args

def preprocess(src_path, save_path):
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(  # datasets.cityscapes()
                        root=src_path,
                        list_path=config.DATASET.TEST_SET,
                        num_samples=config.TEST.NUM_SAMPLES,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        downsample_rate=1)
    for image, _, _, name in tqdm(test_dataset):
        image = np.array(image).astype(np.float32)
        # print(image.shape)
        image.tofile(os.path.join(save_path, name + '.bin'))


if __name__ == '__main__':
    args = parse_args()
    src_path = args.src_path
    save_path = args.save_path
    src_path = os.path.realpath(src_path)
    save_path = os.path.realpath(save_path)
    preprocess(src_path, save_path)
