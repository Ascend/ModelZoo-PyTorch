#!/bin/bash
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

currtime=`date +%Y%m%d%H%M%S`
source scripts/set_npu_env.sh
nohup python3.7  validate_ghostnet.py  \
         /opt/npu/dataset/imagenet/val/  \
        --model GhostNet  \
        -b 512   \
        -j 8 \
        --drop 0.2  \
        --amp  \
        --num-classes 1000 \
        --pretrained > validate_GhostNet_1p.log 2>&1 & 
