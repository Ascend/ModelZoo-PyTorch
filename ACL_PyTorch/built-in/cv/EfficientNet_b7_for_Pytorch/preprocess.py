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
import PIL
import numpy as np
import argparse

import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def preprocess(dataset_path, data_bin_path, batch_size, image_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    val_transforms = transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(dataset_path, val_transforms),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    if not os.path.isdir(data_bin_path):
        os.mkdir(data_bin_path)

    with open('label.txt', 'w+') as f:
        for i, (images, target) in enumerate(data_loader):
            label = ' '.join((str(i) for i in target.tolist()))
            f.write(label+'\n')
            save_file_name = "%d.bin" % i
            save_path = "%s/%s" % (data_bin_path, save_file_name)
            if images.shape[0] != batch_size:
                images = F.pad(input=images, pad=(0, 0, 0, 0, 0, 0, 0, batch_size-images.shape[0]), mode='constant', value=0)
            images.numpy().tofile(save_path)

if __name__ == "__main__":
    """
    python3.7 preprocess.py \
            --dataset_path=data/val \
            --save_path=data_bin \
            --batch_size=16 \
            --image_size=600
    """

    parser = argparse.ArgumentParser(description='EfficientNet preprocess')
    parser.add_argument('--dataset_path', type=str, help='dataset path', required=True)
    parser.add_argument('--save_path', type=str, help='bin file save path', required=True)
    parser.add_argument('--batch_size', type=int, default=16, help='om batch size')
    parser.add_argument('--image_size', type=int, default=600, help='image size')
    args = parser.parse_args()

    preprocess(args.dataset_path, args.save_path, args.batch_size, args.image_size)