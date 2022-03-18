#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
"""
@author: Wenbo Li
@contact: fenglinglwb@gmail.com
"""

import math

import torch
import torchvision.transforms as transforms

from cvpack.dataset import torch_samplers

from dataset.attribute import load_dataset
from dataset.COCO.coco import COCODataset
from dataset.MPII.mpii import MPIIDataset


def get_train_loader(
        cfg, num_gpu, is_dist=True, is_shuffle=True, start_iter=0):
    # -------- get raw dataset interface -------- #
    normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    attr = load_dataset(cfg.DATASET.NAME)
    if cfg.DATASET.NAME == 'COCO':
        Dataset = COCODataset 
    elif cfg.DATASET.NAME == 'MPII':
        Dataset = MPIIDataset
    dataset = Dataset(attr, 'train', transform)

    # -------- make samplers -------- #
    if is_dist:
        sampler = torch_samplers.DistributedSampler(
                dataset, shuffle=is_shuffle)
    elif is_shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)

    images_per_gpu = cfg.SOLVER.IMS_PER_GPU
    # images_per_gpu = cfg.SOLVER.IMS_PER_BATCH // num_gpu

    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []
    if aspect_grouping:
        batch_sampler = torch_samplers.GroupedBatchSampler(
                sampler, dataset, aspect_grouping, images_per_gpu,
                drop_uneven=False)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
                sampler, images_per_gpu, drop_last=False)

    batch_sampler = torch_samplers.IterationBasedBatchSampler(
            batch_sampler, cfg.SOLVER.MAX_ITER, start_iter)

    # -------- make data_loader -------- #
    class BatchCollator(object):
        def __init__(self, size_divisible):
            self.size_divisible = size_divisible

        def __call__(self, batch):
            transposed_batch = list(zip(*batch))
            images = torch.stack(transposed_batch[0], dim=0)
            valids = torch.stack(transposed_batch[1], dim=0)
            labels = torch.stack(transposed_batch[2], dim=0)

            return images, valids, labels

    data_loader = torch.utils.data.DataLoader(
            dataset, num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY), )

    return data_loader


def get_test_loader(cfg, num_gpu, local_rank, stage, is_dist=True):
    # -------- get raw dataset interface -------- #
    normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    attr = load_dataset(cfg.DATASET.NAME)
    if cfg.DATASET.NAME == 'COCO':
        Dataset = COCODataset 
    elif cfg.DATASET.NAME == 'MPII':
        Dataset = MPIIDataset
    dataset = Dataset(attr, stage, transform)

    # -------- split dataset to gpus -------- #
    num_data = dataset.__len__()
    num_data_per_gpu = math.ceil(num_data / num_gpu)
    st = local_rank * num_data_per_gpu
    ed = min(num_data, st + num_data_per_gpu)
    indices = range(st, ed)
    subset= torch.utils.data.Subset(dataset, indices)

    # -------- make samplers -------- #
    sampler = torch.utils.data.sampler.SequentialSampler(subset)

    images_per_gpu = cfg.TEST.IMS_PER_GPU

    batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_gpu, drop_last=False)

    # -------- make data_loader -------- #
    class BatchCollator(object):
        def __init__(self, size_divisible):
            self.size_divisible = size_divisible

        def __call__(self, batch):
            transposed_batch = list(zip(*batch))
            images = torch.stack(transposed_batch[0], dim=0)
            scores = list(transposed_batch[1])
            centers = list(transposed_batch[2])
            scales = list(transposed_batch[3])
            image_ids = list(transposed_batch[4])

            return images, scores, centers, scales, image_ids 

    data_loader = torch.utils.data.DataLoader(
            subset, num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY), )
    data_loader.ori_dataset = dataset

    return data_loader
