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
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch

from timm.data import Dataset, create_loader, resolve_data_config
from timm.utils import *

torch.multiprocessing.set_sharing_strategy('file_system')


def _parse_args():

    parser = argparse.ArgumentParser(description='T2T-ViT preprocess.')
    parser.add_argument('--data-dir', type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('--out-dir', type=str, metavar='PATH', help='path to eval checkpoint')
    parser.add_argument('--gt-path', type=str, metavar='PATH', help='path to groundtruth')
    args = parser.parse_args()

    args.prefetcher = True
    args.distributed = False
    args.batch_size = 1
    args.num_classes = 1000

    return args


def load_val_data(args):

    data_config = resolve_data_config(vars(args), verbose=True)
    dataset_eval = Dataset(args.data_dir)
    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=8,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=False,
    )

    return loader_eval


def pre_process(loader, output_dir, gt_path, args):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    for batch_idx, (input, target) in tqdm(enumerate(loader)):
        if target.shape[0] == args.batch_size:
            bin_data = input.numpy()
            save_path = output_dir / f"{batch_idx:0>5d}.bin"
            bin_data.tofile(save_path)
            labels.append(target)

    np.save(gt_path, np.vstack(labels))


def main():
    args = _parse_args()
    loader = load_val_data(args)
    pre_process(loader, args.out_dir, args.gt_path, args)


if __name__ == '__main__':
    main()
