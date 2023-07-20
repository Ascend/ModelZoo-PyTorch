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

import torch
import timm
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from timm.data import ImageDataset, create_loader, resolve_data_config

def preprocess(dataset_path, save_path, aipp_save_path):

    model = timm.create_model('efficientnetv2_rw_t', pretrained=True)
    config = resolve_data_config({}, model=model)

    if not os.path.isdir(aipp_save_path):
        os.mkdir(aipp_save_path, 0o640)

    torch.multiprocessing.set_sharing_strategy('file_system')

    # process and save aipp data
    for subdir in tqdm(os.listdir(dataset_path)):
        for f in os.listdir(os.path.join(dataset_path, subdir)):
            img_path = os.path.join(dataset_path, subdir, f)
            img = Image.open(img_path).convert('RGB')
            trans_list = [
                transforms.Resize(288, interpolation=Image.BICUBIC),
                transforms.CenterCrop((288,288)),
            ]
            trans = transforms.Compose(trans_list)
            img = trans(img)
            img = np.array(img, dtype=np.uint8)
            img.tofile(os.path.join(aipp_save_path, f'{f}.bin'))

    # process and save normal data for quantization
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

    if not os.path.isdir(save_path):
        os.mkdir(save_path, 0o640)

    for i, (data, _) in enumerate(tqdm(loader)):
        data.numpy().tofile(os.path.join(save_path, f'{i}.bin'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EfficientNetV2 preprocess')
    parser.add_argument('--dataset_path', type=str, help='dataset directory path', required=True)
    parser.add_argument('--save_path', type=str, default='bin_data', 
                        help='save path of preprocessed bin file, default is ./bin_data')
    parser.add_argument('--aipp_save_path', type=str, default='aipp_bin_data', 
                        help='save path of preprocessed bin file for AIPP, default is ./aipp_bin_data')
    args = parser.parse_args()

    preprocess(args.dataset_path, args.save_path, args.aipp_save_path)