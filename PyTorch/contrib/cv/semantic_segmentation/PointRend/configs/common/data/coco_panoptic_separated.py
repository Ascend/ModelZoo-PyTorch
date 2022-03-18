# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2020 Huawei Technologies Co., Ltd
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

from detectron2.config import LazyCall as L
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
)

from .coco import dataloader

dataloader.train.dataset.names = "coco_2017_train_panoptic_separated"
dataloader.train.dataset.filter_empty = False
dataloader.test.dataset.names = "coco_2017_val_panoptic_separated"


dataloader.evaluator = [
    L(COCOEvaluator)(
        dataset_name="${...test.dataset.names}",
    ),
    L(SemSegEvaluator)(
        dataset_name="${...test.dataset.names}",
    ),
    L(COCOPanopticEvaluator)(
        dataset_name="${...test.dataset.names}",
    ),
]
