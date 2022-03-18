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
import numpy as np
import time

import cfg
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class RawDataset(Dataset):

    def __init__(self, is_val=False):
        self.img_h, self.img_w = cfg.max_train_img_size, cfg.max_train_img_size
        if is_val:
            with open(os.path.join(cfg.data_dir, cfg.val_fname), 'r') as f_val:
                f_list = f_val.readlines()
        else:
            with open(os.path.join(cfg.data_dir, cfg.train_fname), 'r') as f_train:
                f_list = f_train.readlines()

        self.image_path_list = []
        self.labels_path_dic = {}
        self.gt_xy_list_path_dic = {}
        for f_line in f_list:
            img_filename = str(f_line).strip().split(',')[0]
            img_path = os.path.join(cfg.data_dir, cfg.train_image_dir_name, img_filename)
            self.image_path_list.append(img_path)
            gt_file = os.path.join(cfg.data_dir, cfg.train_label_dir_name, img_filename[:-4] + '_gt.npy')
            gt_xy_list = os.path.join(cfg.data_dir, cfg.train_label_dir_name, img_filename[:-4] + '.npy')
            self.labels_path_dic[img_path] = gt_file
            self.gt_xy_list_path_dic[img_path] = gt_xy_list
        self.image_path_list.sort()
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        img_path = self.image_path_list[index]
        label = np.load(self.labels_path_dic[img_path])
        gt_xy_list = np.load(self.gt_xy_list_path_dic[img_path])
        try:
            img = Image.open(img_path).convert('RGB')  # for color image

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            img = Image.new('RGB', (self.img_w, self.img_h))
        img_tensor = transforms.ToTensor()(img)
        label = np.transpose(label, (2, 0, 1))

        return (img_tensor, label, gt_xy_list)


def data_collate(batch):
    imgs = []
    labels = []
    gt_xy_list = []  # 长度为N的列表，每个值为该图片中所有矩形框的坐标
    # 例如：[(31, 4, 2), (10, 4, 2), (47, 4, 2), (28, 4, 2)]
    for info in batch:
        imgs.append(info[0])
        labels.append(info[1])
        gt_xy_list.append(info[2])
    return torch.stack(imgs, 0), torch.tensor(np.array(labels)), gt_xy_list


if __name__ == '__main__':
    tick = time.time()
    train_dataset = RawDataset(is_val=False)
    data_loader_A = torch.utils.data.DataLoader(
                train_dataset, batch_size=cfg.batch_size,
                collate_fn=data_collate,
                shuffle=True,
                num_workers=int(cfg.workers),
                pin_memory=True)
    for i, (image_tensors, labels, gt_xy_list) in enumerate(data_loader_A):
        print(image_tensors.shape, labels.shape)
    tock = time.time()
    print(tock-tick)

