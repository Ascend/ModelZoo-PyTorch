#!/usr/bin/env bash

data_path="/opt/npu/"

for para in $*
do
    if [[ $para == --data_path* ]]; then
        data_path=`echo ${para#*=}`
    fi
done

currentDir=$(cd "$(dirname "$0")";pwd)

source ${currentDir}/env_npu.sh

python3 -u ${currentDir}/../train.py --data_root ${data_path} --cfg-options total_epochs=1 \
--resume-from . > ${currentDir}/../i3d_performance_1p.log 2>&1 &
