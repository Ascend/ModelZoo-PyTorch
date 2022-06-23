#!/bin/bash

data_path_info=$1
data_path=`echo ${data_path_info#*=}`
output_dir="output_finetune_8p"
finetune_pth="/home/yzq/MAE/checkpoint-399.pth"

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --finetune_pth* ]];then
        finetune_pth=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#################创建日志输出目录，不需要修改#################
if ! [ -d ${output_dir} ];then
    mkdir -p ${output_dir}
fi

# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source test/env_npu.sh
fi

nohup python -u -m torch.distributed.launch --nproc_per_node=8 --master_port 7164 main_finetune.py \
             --data_path ${data_path} \
             --finetune ${finetune_pth} \
             --output_dir ${output_dir} \
             --model vit_base_patch16 \
             --epochs 1 \
             --world_size 8 \
             --batch_size 256 \
             --num_workers 32 \
             --blr 10e-4 \
             --layer_decay 0.65 \
             --weight_decay 0.05 \
             --drop_path 0.1 \
             --mixup 0.8 \
             --cutmix 1.0 \
             --reprob 0.25 \
             --dist_eval \
             --amp \
             > ${output_dir}/910A_8p_finetune.log 2>&1 &