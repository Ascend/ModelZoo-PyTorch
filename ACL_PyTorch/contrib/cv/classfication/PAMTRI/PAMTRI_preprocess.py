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


from __future__ import print_function
from __future__ import division
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torchreid.data_manager import DatasetManager
from torchreid.dataset_loader import ImageDataset
from torchreid import transforms as T
from torchreid import models
from PIL import Image
import sys

def preprocess():
    print("==========\nArgs:{}\n==========".format(args))
    print("Initializing dataset {}".format(args.dataset))
    dataset = DatasetManager(dataset_dir=args.dataset, root=args.root)

    transform_test = T.Compose_Keypt([
        T.Resize_Keypt((256, 256)),
        T.ToTensor_Keypt(),
        T.Normalize_Keypt(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    queryloader = DataLoader(
        ImageDataset(dataset.query, keyptaware=False, heatmapaware=False, segmentaware=False,
                     transform=transform_test, imagesize=(256, 256)),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=False, drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, keyptaware=False, heatmapaware=False, segmentaware=False,
                     transform=transform_test, imagesize=(256, 256)),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=False, drop_last=False,
    )

    get_bin(queryloader, galleryloader)


def get_bin(queryloader, galleryloader):
    queryloader_num = 0
    galleryloader_num = 0
    for batch_idx, (imgs, vids, camids, vcolors, vtypes, vkeypts) in enumerate(queryloader):
        query = imgs.numpy()
        query.tofile(os.path.join(args.save_path1, "{}.bin".format(queryloader_num)))
        queryloader_num = queryloader_num + 1 

    for batch_idx, (imgs, vids, camids, vcolors, vtypes, vkeypts) in enumerate(galleryloader):
        gallery = imgs.numpy()
        gallery.tofile(os.path.join(args.save_path2, "{}.bin".format(galleryloader_num)))
        galleryloader_num = galleryloader_num + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_dir", default="./data/veri/image_query")
    parser.add_argument("--gallery_dir", default="./data/veri/image_test")
    parser.add_argument("--save_path1", default="./prep_dataset_query")
    parser.add_argument("--save_path2", default="./prep_dataset_gallery")
    parser.add_argument('--root', type=str, default='data',
                        help="root path to data directory")
    parser.add_argument('-d', '--dataset', type=str, default='veri',
                        help="name of the dataset")
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument('--test-batch', default=1, type=int,
                        help="test batch size")
    parser.add_argument('-a', '--arch', type=str, default='densenet121', choices=models.get_names())
    args = parser.parse_args()

    if not os.path.isdir(args.save_path1):
        os.makedirs(os.path.realpath(args.save_path1))
    if not os.path.isdir(args.save_path2):
        os.makedirs(os.path.realpath(args.save_path2))
    preprocess()
