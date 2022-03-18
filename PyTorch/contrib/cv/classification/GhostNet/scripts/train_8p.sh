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

source scripts/set_npu_env.sh

RANK_ID_START=0
RANK_SIZE=8

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do

KERNEL_NUM=$(($(nproc)/8))
PID_START=$((KERNEL_NUM * RANK_ID))
PID_END=$((PID_START + KERNEL_NUM - 1))

nohup \
taskset -c $PID_START-$PID_END python3.7 -u train_ghostnet_8p.py "$@"  \
         /opt/npu/dataset/imagenet \
	--model GhostNet \
        -b 1024   \
	--weight-decay 4e-5 \
	--momentum 0.9 \
        --sched cosine \
	--epochs 400 \
        -j $(($(nproc)/8)) \
        --warmup-lr 1e-6  \
        --drop 0.2  \
        --warmup-epochs 4 \
        --amp  \
        --lr 3.2  \
	--opt npufusedsgd \
        --clip-grad 2.0 \
        --num-classes 1000 \
        --world_size 8 \
	--distributed \
	--use-multi-epochs-loader  \
	--no-prefetcher \
	--local_rank $RANK_ID  & 
done
