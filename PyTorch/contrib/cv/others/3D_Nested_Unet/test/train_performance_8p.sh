#!/bin/bash
# train_performance_8p.sh

# 启动本脚本的示例: bash train_performance_8p.sh
# 单独调用nnUNet_train的示例: python -m torch.distributed.launch --master_port=1234 --nproc_per_node=8 run_training_DDP.py 3d_fullres nnUNetPlusPlusTrainerV2_hypDDP 003 1 --dbs

##########基础配置及超参数##########

export WORLD_SIZE=8
# 可改参数。输出的性能日志文件的路径。
output_log="./test/3D_Nested_Unet_npu_8p_perf.log"
# 注：本模型的较多网络模型参数，均为固定值，不需要修改。
# 注：本模型的其他超参数，例如batchsize和lr，请参考README中的讲解进行修改。
mkdir -p test

##########帮助信息，不需要修改##########

# None

##########参数获取，不需要修改##########

# None

##########参数检验，不需要修改##########

# None

##########启动模型，开始测试FPS##########

date
start_time=$(date +%s)
echo "train_performance_8p start."
python -m torch.distributed.launch --master_port=1234 --nproc_per_node=8 UNetPlusPlus/pytorch/nnunet/run/run_training_DDP.py 3d_fullres nnUNetPlusPlusTrainerV2_hypDDP 003 1 --dbs --other_use fps > ${output_log} 2>&1 &
wait
python -m torch.distributed.launch --master_port=1234 --nproc_per_node=8 UNetPlusPlus/pytorch/nnunet/run/run_training_DDP.py 3d_fullres nnUNetPlusPlusTrainerV2_hypDDP 003 1 --dbs --other_use fps
echo "train_performance_8p.sh end."
date
end_time=$(date +%s)
used_time=$(( $end_time - $start_time ))
echo "total time used(s) : $used_time"
echo "We are going to grep the last 10 lines in the result log file..."
tail -n 10 ${output_log}


