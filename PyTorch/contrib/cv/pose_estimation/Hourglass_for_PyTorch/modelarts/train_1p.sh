#!/usr/bin/env bash
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ============================================================================

ckpt_interval=$1
eval_interval=$2
lr=$3
warmup_iters=&4
warmup_ratio=$5
total_epochs=$6
work_dir=$7
data_root=$8
config=$9

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=-29500 \
    $(dirname "$0")/train.py $ckpt_interval $eval_interval $lr $warmup_iters \
    $warmup_ratio $total_epochs $work_dir $data_root $config

