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

python3 ./train_8p.py \
	--device='npu' \
	--num_points=2500 \
	--workers=128 \
	--nepoch=80 \
	--amp_mode=$true \
	--store_prof=$true \
	--nodes=1 \
	--gpus=8 \
	--dataset=${data_path} \
	--batchSize=1024 > ./pointnet_full_8p.log 2>&1 &
	
