#!/bin/bash

source ./test/env_npu.sh

# 数据集路径,保持为空,不需要修改
data_path=""
# checkpoint文件路径,以实际路径为准
pth_path=""

for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --pth_path* ]];then
        pth_path=`echo ${para#*=}`
    fi
done


if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi


if [[ $pth_path == "" ]];then
    echo "[Error] para \"pth_path\" must be confing"
    exit 1
fi


batch_size=512

python3 tools/run_net.py --cfg demo/Kinetics/X3D_S.yaml NUM_GPUS 8 RNG_SEED 3 TRAIN.ENABLE False TEST.ENABLE False TEST.BATCH_SIZE ${batch_size} TEST.CHECKPOINT_FILE_PATH ${pth_path} DATA.PATH_TO_DATA_DIR ${data_path} DATA_LOADER.NUM_WORKERS 3 OUTPUT_DIR '.'
