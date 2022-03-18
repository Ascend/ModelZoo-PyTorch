#!/bin/bash
export WORLD_SIZE=8
export MASTER_ADDR=localhost
export MASTER_PORT=25684

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi
source ${test_path_dir}/env_npu.sh

for((RANK_ID=0;RANK_ID<WORLD_SIZE;RANK_ID++));
do
    export RANK_ID=$RANK_ID
    if [ $(uname -m) = "aarch64" ]
    then
    KERNEL_NUM=$(($(nproc)/8))
    PID_START=$((KERNEL_NUM * RANK_ID))
    PID_END=$((PID_START + KERNEL_NUM - 1))
    nohup taskset -c $PID_START-$PID_END python3 -u trainer.py \
                    --dataset div2k \
                    --eval_datasets div2k \
                    --model wdsr \
                    --train_batch_size 128 \
                    --scale 2 \
                    --save_checkpoints_epochs 5 \
                    --local_rank $RANK_ID \
                    --job_dir ./wdsr_x2  &
    else
          nohup python3 -u trainer.py \
                    --dataset div2k \
                    --eval_datasets div2k \
                    --model wdsr \
                    --train_batch_size 128 \
                    --scale 2 \
                    --save_checkpoints_epochs 5 \
                    --local_rank $RANK_ID \
                    --job_dir ./wdsr_x2  &
    fi

done
