# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import munch 
import torch
import torch.utils.data as torchdata
import torch.utils.data.distributed as datadist
import torchvision.datasets as datasets
import torchvision.transforms as transforms 


def create_train_loader(model, train_dir, args, pretrained_setting, distributed=False):
    train_dataset = datasets.ImageFolder(train_dir, transforms.Compose([
        transforms.RandomResizedCrop(max(model.input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=pretrained_setting['mean'], 
            std=pretrained_setting['std']
        )
    ]))
    if distributed:
        train_sampler = datadist.DistributedSampler(train_dataset, shuffle=True) 
    else:
        train_sampler = torchdata.RandomSampler(train_dataset)

    train_loader = torchdata.DataLoader(
        train_dataset,
        sampler=train_sampler,
        drop_last=True,
        shuffle=False,
        batch_size=args.batch_size, # // len(args.cuda_ids),
        num_workers=args.num_workers,
        pin_memory=True
    )
    return train_sampler, train_loader


def create_val_loader(model, val_dir, args, scale, distributed=False):
    val_tf = TransformImage(
        model, 
        scale=scale,
        preserve_aspect_ratio=args.preserve_aspect_ratio
    )
    val_dataset = datasets.ImageFolder(val_dir, val_tf)
    if distributed:
        val_sampler = datadist.DistributedSampler(val_dataset, shuffle=False)
    else:
        val_sampler = torchdata.SequentialSampler(val_dataset)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        shuffle=False,
        batch_size=args.batch_size, # // len(args.cuda_ids),
        num_workers=args.num_workers,
        pin_memory=True
    )
    return val_sampler, val_loader


class ToSpaceBGR(object):

    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):

    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor


class TransformImage(object):

    def __init__(self, opts, scale=0.875, random_crop=False,
                 random_hflip=False, random_vflip=False,
                 preserve_aspect_ratio=True):
        if type(opts) == dict:
            opts = munch.munchify(opts)
        self.input_size = opts.input_size
        self.input_space = opts.input_space
        self.input_range = opts.input_range
        self.mean = opts.mean
        self.std = opts.std

        # https://github.com/tensorflow/models/blob/master/research/inception/inception/image_processing.py#L294
        self.scale = scale
        self.random_crop = random_crop
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip

        tfs = []
        if preserve_aspect_ratio:
            tfs.append(transforms.Resize(int(math.floor(max(self.input_size)/self.scale))))
        else:
            height = int(self.input_size[1] / self.scale)
            width = int(self.input_size[2] / self.scale)
            tfs.append(transforms.Resize((height, width)))

        if random_crop:
            tfs.append(transforms.RandomCrop(max(self.input_size)))
        else:
            tfs.append(transforms.CenterCrop(max(self.input_size)))

        if random_hflip:
            tfs.append(transforms.RandomHorizontalFlip())

        if random_vflip:
            tfs.append(transforms.RandomVerticalFlip())

        tfs.append(transforms.ToTensor())
        tfs.append(ToSpaceBGR(self.input_space=='BGR'))
        tfs.append(ToRange255(max(self.input_range)==255))
        tfs.append(transforms.Normalize(mean=self.mean, std=self.std))

        self.tf = transforms.Compose(tfs)

    def __call__(self, img):
        tensor = self.tf(img)
        return tensor
