#!/bin/bash
source set_env.sh

# 参数校验
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

# 校验是否传入pth_path,不需要修改
if [[ $pth_path == "" ]];then
    echo "[Error] para \"pth_path\" must be confing"
    exit 1
fi

# 校验是否传入num_classes,不需要修改
if [[ $num_classes == "" ]];then
    echo "[Error] para \"num_classes\" must be confing"
    exit 1
fi

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

python3 ./finetune.py \
	--num_classes=${num_classes} \
	--model=${pth_path} \
	--nepoch=1 \
	--workers=32 \
	--num_points=2500 \
	--dataset=${data_path} \
	--batchSize=64 > ./pointnet_finetune_1p.log 2>&1 &
