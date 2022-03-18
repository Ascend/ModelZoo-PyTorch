#!/usr/bin/env bash
currentDir=$(cd "$(dirname "$0")";pwd)
echo 'Current directory is: '$currentDir
source $currentDir'/npu_set_env.sh'
export datasetPath=$currentDir'/../CASIA-B-Pre'

N_NPUS=$(python3.7.5 -c """
from config import conf_8p as conf
device_str = conf['ASCEND_VISIBLE_DEVICES']
print(len(device_str) // 2 + 1)"""
)
echo 'Using '$N_NPUS' NPUs...'

nohup python -m torch.distributed.launch --nproc_per_node=$N_NPUS train_main.py \
    --dist_backend='hccl' \
    --world_size=$N_NPUS \
    --rank=0 \
    --device_num=$N_NPUS | tee $currentDir'/../logTrain8p.txt' &
