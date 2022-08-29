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
# ============================================================================
import os
import re
import sys
from tqdm import tqdm
from time import time
sys.path.append('./')
# general libs
import logging
import numpy as np

# pytorch libs
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# densetorch wrapper
import densetorch as dt

# configuration for light-weight refinenet
from arguments import get_arguments
from data import get_datasets, get_transforms
from network import get_segmenter
from optimisers import get_optimisers, get_lr_schedulers
from apex import amp
import torch.multiprocessing as mp

def setup_data_loaders(args):
    train_transforms, val_transforms = get_transforms(
        crop_size=args.crop_size,
        shorter_side=args.shorter_side,
        low_scale=args.low_scale,
        high_scale=args.high_scale,
        img_mean=args.img_mean,
        img_std=args.img_std,
        img_scale=args.img_scale,
        ignore_label=args.ignore_label,
        num_stages=args.num_stages,
        augmentations_type=args.augmentations_type,
        dataset_type=args.dataset_type,
    )
    train_sets, val_set = get_datasets(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        train_list_path=args.train_list_path,
        val_list_path=args.val_list_path,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        masks_names=("segm",),
        dataset_type=args.dataset_type,
        stage_names=args.stage_names,
        train_download=args.train_download,
        val_download=args.val_download,
    )


args = get_arguments()
setup_data_loaders(args)

