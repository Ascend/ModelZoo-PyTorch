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

import os
import argparse

from tqdm import tqdm
import torch
import timm
from timm.data import ImageDataset, create_loader, resolve_data_config

def preprocess(dataset_path, data_save_path):

    model = timm.create_model('efficientnetv2_rw_t', pretrained=True)
    config = resolve_data_config({}, model=model)

    torch.multiprocessing.set_sharing_strategy('file_system')

    loader = create_loader(
            ImageDataset(dataset_path),
            input_size=(3, 288, 288),
            batch_size=1,
            use_prefetcher=False,
            interpolation=config['interpolation'],
            mean=config['mean'],
            std=config['std'],
            num_workers=2,
            crop_pct=config['crop_pct']
    )

    if not os.path.isdir(data_save_path):
        os.mkdir(data_save_path)

    labels = []
    for i, (data, label) in enumerate(tqdm(loader)):
        labels.append(' '.join([str(i) for i in label.numpy()]))
        data.numpy().tofile(os.path.join(data_save_path, f'{i}.bin'))

    with open('label.txt', 'w') as f:
        for label in labels:
            f.write(label + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EfficientNetV2 preprocess')
    parser.add_argument('--dataset_path', type=str, help='dataset path', required=True)
    parser.add_argument('--data_save_path', type=str, help='bin file save path', required=True)
    args = parser.parse_args()

    preprocess(args.dataset_path, args.data_save_path)