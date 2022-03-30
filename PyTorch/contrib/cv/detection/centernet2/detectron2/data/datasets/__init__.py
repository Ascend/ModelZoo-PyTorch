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
#
#
# Copyright (c) Facebook, Inc. and its affiliates.
from .coco import load_coco_json, load_sem_seg, register_coco_instances
from .coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
from .lvis import load_lvis_json, register_lvis_instances, get_lvis_instances_meta
from .pascal_voc import load_voc_instances, register_pascal_voc
from . import builtin as _builtin  # ensure the builtin datasets are registered


__all__ = [k for k in globals().keys() if not k.startswith("_")]
