#!/bin/bash
# train_full_8p.sh

# 启动本脚本的示例: bash train_full_8p.sh -path /home/heyupeng/environment/
# 单独调用nnUNet_train的示例: python -m torch.distributed.launch --master_port=1234 --nproc_per_node=8 run_training_DDP.py 3d_fullres nnUNetPlusPlusTrainerV2_hypDDP 003 0 --dbs
# 单独调用get_dice_result.py的示例: python get_dice_result.py -path /home/heyupeng/environment/ -mode 8p

##########基础配置及超参数##########

export WORLD_SIZE=8
# 必改参数。指向了RESULTS_FOLDER文件夹所在目录。
path="/home/heyupeng/environment/"
# 可改参数。输出的训练日志文件的路径。
output_log1="./test/3D_Nested_Unet_npu_8p_train.log"
# 可改参数。输出的精度日志文件的路径。
output_log2="./test/3D_Nested_Unet_npu_8p_dice.log"
# 注：本模型的较多网络模型参数，均为固定值，不需要修改。
# 注：本模型的其他超参数，例如batchsize和lr，请参考README中的讲解进行修改。
mkdir -p test

##########帮助信息，不需要修改##########

if [[ $1 == --help || $1 == -h ]];then
    echo "train_full_8p.sh"
    echo "parameter explain:
    -val/--validation_only：use this if you want to only run the validation
    -c/--continue_training：use this if you want to continue a training
    -h/--help：show help message
    "
    exit 1
fi

##########参数获取，不需要修改##########

validation_only=""
continue_training=""
for para in $*
do
    if [[ $para == -val* || $para == --validation_only* ]];then
        validation_only=`echo ${para#*=}`
    elif [[ $para == -c* || $para == --continue_training* ]];then
        continue_training=`echo ${para#*=}`
    fi
done

##########参数检验，不需要修改##########

# None

##########启动模型，开始训练##########

date
start_time=$(date +%s)
echo "train_full_8p start."
python -m torch.distributed.launch --master_port=1234 --nproc_per_node=8 UNetPlusPlus/pytorch/nnunet/run/run_training_DDP.py 3d_fullres nnUNetPlusPlusTrainerV2_hypDDP 003 0 --dbs ${validation_only} ${continue_training} > ${output_log1} 2>&1 &
wait
echo "train_full_8p.sh end."
date
end_time=$(date +%s)
used_time=$(( $end_time - $start_time ))
echo "total time used(s) : $used_time"

##########获取最终的精度指标##########
echo "get the result start."
python get_dice_result.py -path ${path} -mode 8p > ${output_log2} 2>&1 &
wait
echo "We are going to grep the last 10 lines in the result log file..."
tail -n 10 ${output_log2}
echo "get the result done."
