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
data_path="/root/data/VOCdevkit"

for para in $*
do
    if [[ $para == --data_path* ]]; then
        data_path=`echo ${para#*=}`
    fi
done

echo ${data_path}


RANK_ID_START=0
RANK_SIZE=8

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do

KERNEL_NUM=8
PID_START=$((KERNEL_NUM * RANK_ID))
PID_END=$((PID_START + KERNEL_NUM - 1))

nohup \
taskset -c $PID_START-$PID_END python -u train_8p.py "$@"  \
    --dataset_root ${data_path} \
    --save_folder ./RefineDet320_bn/ \
    --num_workers $KERNEL_NUM \
    --world_size 8 \
    --batch_size 32 \
    --bn True \
    --local_rank $RANK_ID \
    --lr 0.002 &
done

