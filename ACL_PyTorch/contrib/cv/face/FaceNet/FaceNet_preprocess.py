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
import argparse
import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from facenet_pytorch import fixed_image_standardization
from torch.utils.data import DataLoader, SequentialSampler


def face_preprocess(crop_dir, save_dir):
    # create dataset and data loaders from cropped images output from MTCNN
    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    dataset = datasets.ImageFolder(crop_dir, transform=trans)

    embed_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset)
    )

    for idx, (xb, yb) in tqdm(enumerate(embed_loader)):
        out_path_xb = os.path.join(save_dir, 'xb_results', '{}.bin'.format(idx))
        out_path_yb = os.path.join(save_dir, 'yb_results', '{}.bin'.format(idx))
        os.makedirs(os.path.dirname(out_path_xb), exist_ok=True)
        os.makedirs(os.path.dirname(out_path_yb), exist_ok=True)
        if xb.shape[0] < batch_size:
            xb_zeros = np.zeros([batch_size - int(xb.shape[0]), int(xb.shape[1]), int(xb.shape[2]), int(xb.shape[3])])
            xb = np.concatenate([xb.numpy(), xb_zeros], axis=0)
            xb = torch.from_numpy(xb)
        xb.detach().cpu().numpy().tofile(out_path_xb)
        yb.detach().cpu().numpy().tofile(out_path_yb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_dir', type=str, help='cropped image save path')
    parser.add_argument('--save_dir', type=str, help='preprocess bin files save path')
    parser.add_argument('--batch_size', type=int, default=1, help='preprocess bin files save path')
    arg = parser.parse_args()
    batch_size = arg.batch_size
    workers = 0 if os.name == 'nt' else 8
    face_preprocess(arg.crop_dir, arg.save_dir)
