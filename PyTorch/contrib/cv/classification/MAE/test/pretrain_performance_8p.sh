#!/bin/bash

# 训练使用的npu卡数
export RANK_SIZE=8
export WORLD_SIZE=8
data_path_info=$1
data_path=`echo ${data_path_info#*=}`
output_dir="output_pretrain_8p"

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
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

nohup python -u -m torch.distributed.launch --nproc_per_node=8 --master_port 7164 main_pretrain.py \
             --data_path ${data_path} \
             --model mae_vit_base_patch16 \
             --epochs 1 \
             --world_size 8 \
             --batch_size 256 \
             --num_workers 32 \
             --blr 3e-4 \
             --norm_pix_loss \
             --amp \
             --output_dir ${output_dir} \
             > ${output_dir}/910A_8p_pretrain.log 2>&1 &