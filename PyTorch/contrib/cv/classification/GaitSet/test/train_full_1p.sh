#!/usr/bin/env bash
currentDir=$(cd "$(dirname "$0")";pwd)
echo 'Current directory is: '$currentDir
source $currentDir'/npu_set_env.sh'
export datasetPath=$currentDir'/../CASIA-B-Pre'

nohup python3.7.5 -u train_main.py \
    --dist_backend='hccl' \
    --world_size=1 \
    --rank=0 \
    --device_num=1 | tee $currentDir'/../consoleLogTrain1p.txt' &
