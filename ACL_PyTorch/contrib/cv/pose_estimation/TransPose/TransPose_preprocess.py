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

import argparse
import sys
import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.append('./TransPose')
from lib.config import cfg
from lib.config import update_config
from lib.dataset.coco import COCODataset

parser = argparse.ArgumentParser(description='Test keypoints network')
# general
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    default="TransPose/experiments/coco/transpose_r/TP_R_256x192_d256_h1024_enc3_mh8.yaml",
                    type=str)

parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
parser.add_argument('--output', dest='output',
                    help='output for prepared data', default='prep_data',
                    type=str)
parser.add_argument('--output_flip', dest='output_flip',
                    help='output for prepared flip data', default='prep_data_flip',
                    type=str)

opt = parser.parse_args()

os.makedirs(opt.output)
os.makedirs(opt.output_flip)


def preprocess(config):
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = COCODataset(
        config, config.DATASET.ROOT, config.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    for idx, (image, _, _, _) in tqdm(enumerate(valid_loader)):
        image_flip = torch.flip(image, [3])
        image = image.numpy()
        image_flip = image_flip.numpy()

        # print(image.shape)
        output_name = "{:0>12d}.bin".format(idx)
        output_path = os.path.join(opt.output, output_name)
        image.tofile(output_path)

        output_name_flip = "{:0>12d}.bin".format(idx)
        output_path_flip = os.path.join(opt.output_flip, output_name_flip)
        image_flip.tofile(output_path_flip)


if __name__ == '__main__':
    update_config(cfg, opt)
    preprocess(cfg)
