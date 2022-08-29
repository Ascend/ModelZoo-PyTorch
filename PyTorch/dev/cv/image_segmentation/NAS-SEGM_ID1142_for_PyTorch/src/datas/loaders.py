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
"""Create PyTorch's DataLoaders"""

import logging

# Torch libraries
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

# Custom libraries
from .datasets import PascalCustomDataset as Dataset
from .datasets import (
    CentralCrop,
    Normalise,
    RandomCrop,
    RandomMirror,
    ResizeScale,
    ToTensor,
)
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def create_loaders(args):
    """
    Args:
      train_dir (str) : path to the root directory of the training set.
      val_dir (str) : path to the root directory of the validation set.
      train_list (str) : path to the training list.
      val_list (str) : path to the validation list.
      meta_train_prct (int) : percentage of meta-train.
      shorter_side (int) : parameter of the shorter_side resize transformation.
      crop_size (int) : square crop to apply during the training.
      normalise_params (list / tuple) : img_scale, img_mean, img_std.
      batch_size (int) : training batch size.
      num_workers (int) : number of workers to parallelise data loading operations.

    If train_list == val_list, then divide train_list into meta-train and meta-val.

    Returns:
      train_loader, val loader, do_search (boolean, train_list == val_list).

    """
    ## Transformations during training ##
    logger = logging.getLogger(__name__)
    composed_trn = transforms.Compose(
        [
            ResizeScale(
                args.resize_side[0],
                args.low_scale,
                args.high_scale,
                args.resize_longer_side,
            ),
            RandomMirror(),
            RandomCrop(args.crop_size[0]),
            Normalise(*args.normalise_params),
            ToTensor(),
        ]
    )
    composed_val = transforms.Compose(
        [
            ResizeScale(args.val_resize_side, 1, 1, args.resize_longer_side),
            CentralCrop(args.val_crop_size),
            Normalise(*args.normalise_params),
            ToTensor(),
        ]
    )
    ## Training and validation sets ##
    trainset = Dataset(
        data_file=args.train_list,
        data_dir=args.train_dir,
        transform_trn=composed_trn,
        transform_val=composed_val,
    )
    do_search = False
    if args.train_list == args.val_list:
        do_search = True
        # Split train into meta-train and meta-val
        n_examples = len(trainset)
        n_train = int(n_examples * args.meta_train_prct / 100.0)
        trainset, valset = random_split(trainset, [n_train, n_examples - n_train])
    else:
        valset = Dataset(
            data_file=args.val_list,
            data_dir=args.val_dir,
            transform_trn=None,
            transform_val=composed_val,
        )
    logger.info(
        " Created train set = {} examples, val set = {} examples; do_search = {}".format(
            len(trainset), len(valset), do_search
        )
    )
    ## Training and validation loaders ##
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size[0],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        valset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, val_loader, do_search
