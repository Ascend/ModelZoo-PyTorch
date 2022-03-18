#!/bin/bash

# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

echo "Please run the script as: "
echo "sh test/run8p.sh DATA_PATH"
echo "for example: sh test/run8p.sh /dataset_path "
echo "After running the script, the network runs in the background, The log will be generated in ./train_0.log"

source ./test/env.sh
cur_path=`pwd`
export PYTHONPATH=$cur_path/../WDL_for_PyTorch:$PYTHONPATH
export RANK_SIZE=8

DATA_PATH=$1

start_time=$(date +%s)
output_dir=${cur_path}/output_8p_${start_time}
if [ -d ${output_dir}/ ];then
    rm -rf ${output_dir}
    mkdir -p ${output_dir}
else
    mkdir -p ${output_dir}
fi

if [ $(uname -m) = "aarch64" ]
then
    KERNEL_NUM=$(($(nproc)/8))
    for i in $(seq 0 7)
    do
    PID_START=$((KERNEL_NUM * i))
    PID_END=$((PID_START + KERNEL_NUM - 1))
    taskset -c $PID_START-$PID_END $CMD python3 -u run_classification_criteo_wdl.py \
        --device_id $i \
        --data_path $DATA_PATH \
        --checkpoint_save_path=${output_dir} \
        --lr=0.0009 \
	      --sparse_embed_dim 4 \
	      --batch_size 4096 \
	      --epochs 3 \
        --amp \
        --device_num 8 \
        --dist > ${output_dir}/train_$i.log 2>&1 &
    done
else
    for i in $(seq 0 7)
    do
    python3.7 -u run_classification_criteo_wdl.py \
        --device_id $i \
        --data_path $DATA_PATH \
        --checkpoint_save_path=${output_dir} \
        --lr=0.0009 \
	      --sparse_embed_dim 4 \
	      --batch_size 4096 \
	      --epochs 3 \
        --amp \
        --device_num 8 \
        --dist > ${output_dir}/train_$i.log 2>&1 &
    done
fi
