# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import NamedTuple


class ClusterConfig(NamedTuple):
    dist_backend: str
    dist_url: str


class TrainerConfig(NamedTuple):
    data_folder: str
    epochs: int
    lr: float
    input_size: int
    batch_per_gpu: int
    save_folder: str
    imnet_path: str
    workers: int
    local_rank: int
    global_rank: int
    num_tasks: int
    job_id: str
