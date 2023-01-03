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


import sys
import os
sys.path.append('./GMA/core')
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils.utils import InputPadder
from utils import frame_utils


def parser_func():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str, default='./MPI-Sintel-complete/training')
    parser.add_argument('-o', '--output', default='./data_preprocessed_bs1', help='output path.')
    parser.add_argument('-s', '--status', type=str, default='clean')
    args1 = parser.parse_args()
    return args1


def adapt(image1, image2):
    """Preprocess the input samples to adapt them to the network's requirements
    Here, x, is the actual data, not the x TF tensor.
    Args:
        x: input samples in list[(2,H,W,3)] or (N,2,H,W,3) np array form
    Returns:
        Samples ready to be given to the network (w. same shape as x)
        Also, return adaptation info in (N,2,H,W,3) format
    """
    # Ensure we're dealing with RGB image pairs
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    return image1.numpy(), image2.numpy()


def get_data_pairs():
    img_dir = os.path.join(_DATASET_ROOT, status)
    flow_dir = os.path.join(_DATASET_ROOT, 'flow')

    folders = sorted([os.path.basename(_) for _ in os.listdir(img_dir)])
    out_pairs = []
    for folder in folders:
        num_file = len(os.listdir(os.path.join(img_dir, folder)))
        for idx in range(1, num_file):
            pair = (
                os.path.join(img_dir, folder, "frame_{:0>4d}.png".format(idx)),
                os.path.join(img_dir, folder, "frame_{:0>4d}.png".format(idx+1)),
                os.path.join(flow_dir, folder, "frame_{:0>4d}.flo".format(idx)),
            )
            out_pairs.append(pair)
    return out_pairs


if __name__ == '__main__':
    args = parser_func()
    _DATASET_ROOT = args.dataset
    status = args.status

    data_pairs = get_data_pairs()
    val_num = len(data_pairs)

    images1 = []
    images2 = []
    gts = []
    index = 0

    if not os.path.exists(os.path.join(args.output, 'image1')):
        os.makedirs(os.path.join(args.output, 'image1'))
    if not os.path.exists(os.path.join(args.output, 'image2')):
        os.makedirs(os.path.join(args.output, 'image2'))
    if not os.path.exists(os.path.join(args.output, 'gt')):
        os.makedirs(os.path.join(args.output, 'gt'))

    for i, data in tqdm(enumerate(data_pairs), total=len(data_pairs)):
        image1 = Image.open(data[0])
        image1 = np.array(image1)
        image2 = Image.open(data[1])
        image2 = np.array(image2)
        label = frame_utils.read_gen(data[2])
        image1, image2 = adapt(image1, image2)
        gt_label = label
        images1.append(image1)
        images2.append(image2)

        out_path_gt = os.path.join(args.output, 'gt', '{0}.bin'.format(i))
        gt_label.tofile(out_path_gt)

        out_path_image1 = os.path.join(args.output, 'image1', '{0}.bin'.format(index))
        out_path_image2 = os.path.join(args.output, 'image2', '{0}.bin'.format(index))
        np.array(images1).tofile(out_path_image1)
        np.array(images2).tofile(out_path_image2)
        images1.clear()
        images2.clear()
        index += 1
