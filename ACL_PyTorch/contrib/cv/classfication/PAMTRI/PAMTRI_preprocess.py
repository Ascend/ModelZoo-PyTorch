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


from tabnanny import verbose
from torchreid import transforms as T
from torchreid.dataset_loader import ImageDataset
from torchreid.data_manager import DatasetManager
import os
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append('./PAMTRI/MultiTaskNet')


def preprocess(args):
    dataset = DatasetManager(dataset_dir=args.dataset,
                             root=args.root,
                             verbose=False)

    transform_test = T.Compose_Keypt([
        T.Resize_Keypt((256, 256)),
        T.ToTensor_Keypt(),
        T.Normalize_Keypt(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
    ])

    queryloader = DataLoader(
        ImageDataset(dataset.query, keyptaware=False, heatmapaware=False, segmentaware=False,
                     transform=transform_test, imagesize=(256, 256)),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=False, drop_last=False,
    )
    for batch_idx, (query,
                    vids,
                    camids,
                    vcolors,
                    vtypes,
                    vkeypts) in enumerate(tqdm(queryloader,
                                               desc="Preprocessing query data...")):
        query.numpy().tofile(os.path.join(args.save_query, f"{batch_idx}.bin"))

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, keyptaware=False, heatmapaware=False, segmentaware=False,
                     transform=transform_test, imagesize=(256, 256)),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=False, drop_last=False,
    )
    for batch_idx, (gallery,
                    vids,
                    camids,
                    vcolors,
                    vtypes,
                    vkeypts) in enumerate(tqdm(galleryloader,
                                               desc="Preprocessing gallery data...")):
        gallery.numpy().tofile(os.path.join(args.save_gallery,
                                            f"{batch_idx}.bin"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_dir", default="/opt/npu/veri/image_query")
    parser.add_argument("--gallery_dir", default="/opt/npu/veri/image_test")
    parser.add_argument("--save_query", default="./prep_dataset_query")
    parser.add_argument("--save_gallery", default="./prep_dataset_gallery")
    parser.add_argument('--root', type=str, default='./PAMTRI/MultiTaskNet/data',
                        help="root path to data directory")
    parser.add_argument('-d', '--dataset', type=str, default='veri',
                        help="name of the dataset")
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument('--test-batch', default=1, type=int,
                        help="test batch size")
    args = parser.parse_args()

    if not os.path.isdir(args.save_query):
        os.makedirs(os.path.realpath(args.save_query))
    if not os.path.isdir(args.save_gallery):
        os.makedirs(os.path.realpath(args.save_gallery))
    preprocess(args)
