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
import argparse
import tqdm
from pathlib import Path

import torch
import numpy as np

from datasets import build_dataset

torch.multiprocessing.set_sharing_strategy('file_system')


def image2bin(data_root, save_dir, batch_size, data_cfg):

    data_cfg.data_path = data_root
    data_cfg.data_set = 'IMNET'
    data_cfg.color_jitter = 0.4
    data_cfg.aa = 'rand-m9-mstd0.5-inc1'
    data_cfg.train_interpolation = 'bicubic'
    data_cfg.reprob = 0.25
    data_cfg.remode = 'pixel'
    data_cfg.recount = 1
    data_cfg.input_size = 224

    seed = 20200220
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_val, _ = build_dataset(is_train=False, args=data_cfg)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )

    all_labels = []
    for idx, (img_tensor, label_tensor) in enumerate(
                tqdm.tqdm(data_loader_val, desc="Processing")):
        save_path = Path(save_dir) / f'batch-{idx:0>5d}.bin'
        if label_tensor.numel() != batch_size:
            break
        img_tensor.numpy().tofile(save_path)
        all_labels.append(label_tensor.numpy())

    all_labels = np.vstack(all_labels)
    np.save(Path(save_dir).parent / 'labels.npy', all_labels)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('validate model')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--data_root', default='/opt/npu/imagenet/', type=str,
                        help='dataset path')
    parser.add_argument('--save_dir', required=True, type=str,
                        help='dir path to save bin')
    args = parser.parse_args()

    if not Path(args.save_dir).is_dir():
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    image2bin(args.data_root, args.save_dir, args.batch_size, args)
