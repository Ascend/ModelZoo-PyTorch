#!/usr/bin/env bash

data_path="/opt/npu/"

for para in $*
do
    if [[ $para == --data_path* ]]; then
        data_path=`echo ${para#*=}`
    fi
done

gpus=8
port=29111
currentDir=$(cd "$(dirname "$0")";pwd)

source ${currentDir}/env_npu.sh

taskset -c 0-47 python3 -m torch.distributed.launch --nproc_per_node=$gpus --master_port=$port \
    ${currentDir}/../train.py --resume-from . --validate --launcher pytorch \
    --gpu-ids 0 --data_root ${data_path} > ${currentDir}/../i3d_full_8p_new.log 2>&1 &
