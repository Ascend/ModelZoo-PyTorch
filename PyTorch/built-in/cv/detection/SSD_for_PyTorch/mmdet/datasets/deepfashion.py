# Copyright 2022 Huawei Technologies Co., Ltd.
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
# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class DeepFashionDataset(CocoDataset):

    CLASSES = ('top', 'skirt', 'leggings', 'dress', 'outer', 'pants', 'bag',
               'neckwear', 'headwear', 'eyeglass', 'belt', 'footwear', 'hair',
               'skin', 'face')

    PALETTE = [(0, 192, 64), (0, 64, 96), (128, 192, 192), (0, 64, 64),
               (0, 192, 224), (0, 192, 192), (128, 192, 64), (0, 192, 96),
               (128, 32, 192), (0, 0, 224), (0, 0, 64), (0, 160, 192),
               (128, 0, 96), (128, 0, 192), (0, 32, 192)]
