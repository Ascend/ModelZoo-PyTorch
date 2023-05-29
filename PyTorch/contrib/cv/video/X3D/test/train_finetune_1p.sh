#!/bin/bash

source ./test/env_npu.sh
# 数据集路径,保持为空,不需要修改
data_path=""
# checkpoint文件路径,以实际路径为准
pth_path=""
# k400 数据集为 400 分类
num_classes=-1 

for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --pth_path* ]];then
        pth_path=`echo ${para#*=}`
	elif [[ $para == --num_classes* ]];then
        num_classes=`echo ${para#*=}`	
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


if [[ $num_classes == -1 ]];then
    echo "[Error] para \"num_classes\" must be confing"
    exit 1
fi

python3 tools/run_net.py --cfg demo/Kinetics/X3D_S.yaml NUM_GPUS 1 RNG_SEED 3 TRAIN.ENABLE True TRAIN.BATCH_SIZE 32 TRAIN.AUTO_RESUME True TRAIN.CHECKPOINT_FILE_PATH ${pth_path} TRAIN.CHECKPOINT_PERIOD 1 TRAIN.EVAL_PERIOD 1 TEST.ENABLE False TEST.BATCH_SIZE 512 MODEL.NUM_CLASSES ${num_classes} SOLVER.BASE_LR 0.1 SOLVER.MAX_EPOCH 300 SOLVER.WARMUP_START_LR 0.01 DATA.PATH_TO_DATA_DIR ${data_path} DATA_LOADER.NUM_WORKERS 3 APEX.ENABLE True APEX.OPT_LEVEL "O2" APEX.LOSS_SCALE 128.0 DIST_BACKEND 'hccl' OUTPUT_DIR '.'
		            

