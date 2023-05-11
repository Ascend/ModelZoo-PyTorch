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

python3 -u -m torch.distributed.launch --nproc_per_node=$gpus --master_port=$port \
${currentDir}/../test.py --checkpoint ${currentDir}/../result_full/best_top1_acc_epoch_40.pth \
--launcher pytorch --data_root ${data_path} > ${currentDir}/../i3d_eval_8p.log 2>&1 &

