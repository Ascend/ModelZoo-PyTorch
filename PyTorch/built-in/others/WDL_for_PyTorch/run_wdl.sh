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
echo "sh test/run1p.sh DEVICE_ID DATA_PATH"
echo "for example: sh test/run1p.sh 0 /dataset_path"
echo "After running the script, the network runs in the background, The log will be generated in ./train_1p.log"

source ./test/env.sh
cur_path=`pwd`
export PYTHONPATH=$cur_path/../WDL_for_PyTorch:$PYTHONPATH
export RANK_SIZE=1

DEVICE_ID=$1
DATA_PATH=$2

start_time=$(date +%s)
output_dir=${cur_path}/output_1p_${start_time}
if [ -d ${output_dir}/ ];then
    rm -rf ${output_dir}
    mkdir -p ${output_dir}
else
    mkdir -p ${output_dir}
fi

python3 run_classification_criteo_wdl.py  \
    --amp \
    --epochs 3 \
    --checkpoint_save_path=${output_dir} \
    --device_id=$DEVICE_ID \
    --data_path=$DATA_PATH \
    --batch_size 4096 \
    --lr 0.0009 > ${output_dir}/train_1p.log 2>&1 &