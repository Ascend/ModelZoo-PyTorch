#!/usr/bin/env python
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
from .auto_augment import RandAugment, AutoAugment, rand_augment_ops, auto_augment_policy,\
    rand_augment_transform, auto_augment_transform
from .config import resolve_data_config
from .constants import *
from .dataset import ImageDataset, IterableImageDataset, AugMixDataset
from .dataset_factory import create_dataset
from .loader import create_loader
from .mixup import Mixup, FastCollateMixup
from .parsers import create_parser
from .real_labels import RealLabelsImagenet
from .transforms import *
from .transforms_factory import create_transform