#!/bin/bash
source set_env.sh

# 参数校验
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

python3 ./train_1p.py \
	--device='npu' \
	--workers=32 \
	--num_points=2500 \
	--nepoch=2 \
	--store_prof=$false \
	--amp_mode=$true \
	--dataset=${data_path} \
	--batchSize=128 > ./pointnet_performance_1p.log 2>&1 &
