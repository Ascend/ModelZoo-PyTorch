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
import argparse
import sys
from tool import resize_align_multi_scale
sys.path.append('./DEKR')
from tools import _init_paths

from tqdm import tqdm
import torch
import torchvision.transforms
from lib.config import cfg
from lib.config import update_config
from lib.dataset import make_test_dataloader


def preprocess(config):
    scale_list = (512, 768, 1024)
    data_loader, test_dataset = make_test_dataloader(config)
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    pbar = tqdm(total=len(test_dataset), desc='Preprocessing')
    for idx, images in enumerate(data_loader):
        image = images[0].cpu().numpy()
        input_size = config.DATASET.INPUT_SIZE

        image_resized, center, scale = resize_align_multi_scale(
            image, input_size, 1.0, min(config.TEST.SCALE_FACTOR), scale_list
        )

        prefix = 'shape_{}x{}'.format(image_resized.shape[0], image_resized.shape[1])

        output_path = os.path.join(opt.output, prefix)
        output_path_flip = os.path.join(opt.output_flip, prefix)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(output_path_flip):
            os.makedirs(output_path_flip)

        image_resized = transforms(image_resized)
        image_resized = image_resized.unsqueeze(0)
        image_resized_flip = torch.flip(image_resized, [3])
        image_resized_flip[:, :, :, :-3] = image_resized_flip[:, :, :, 3:]

        image_resized = image_resized.numpy()
        image_resized_flip = image_resized_flip.numpy()

        output_name = "{:0>12d}.bin".format(idx)
        output_path = os.path.join(output_path, output_name)
        image_resized.tofile(output_path)

        output_name_flip = "{:0>12d}.bin".format(idx)
        output_path_flip = os.path.join(output_path_flip, output_name_flip)
        image_resized_flip.tofile(output_path_flip)

        pbar.update()

    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test keypoints network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default=
                        "./DEKR/experiments/coco/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_coco_x140.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--output', dest='output',
                        help='output for prepared data', default='prep_data',
                        type=str)
    parser.add_argument('--output_flip', dest='output_flip',
                        help='output for prepared fliped data', default='prep_data_flip',
                        type=str)

    opt = parser.parse_args()
    update_config(cfg, opt)

    preprocess(cfg)