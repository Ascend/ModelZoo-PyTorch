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

from detectron2.data.detection_utils import create_keypoint_hflip_indices

from .coco import dataloader

dataloader.train.dataset.min_keypoints = 1
dataloader.train.dataset.names = "keypoints_coco_2017_train"
dataloader.test.dataset.names = "keypoints_coco_2017_val"

dataloader.train.mapper.update(
    use_instance_mask=False,
    use_keypoint=True,
    keypoint_hflip_indices=create_keypoint_hflip_indices(dataloader.train.dataset.names),
)
