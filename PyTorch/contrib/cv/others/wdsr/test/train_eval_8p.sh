#!/bin/bash
export WORLD_SIZE=8
export MASTER_ADDR=localhost
export MASTER_PORT=25684

pre_train_model=""
for para in $*
do
    if [[ $para == --pre_train_model* ]];then
        pre_train_model=`echo ${para#*=}`
    fi
done

if [[ $pre_train_model == "" ]];then
    echo "[Error] para \"pre_train_model\" must be confing"
    exit 1
fi

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

    nohup python3 -u trainer.py \
    --dataset div2k \
    --ckpt ${pre_train_model} \
    --eval_datasets div2k \
    --model wdsr \
    --train_batch_size 128 \
    --scale 2 \
    --local_rank $RANK_ID \
    --job_dir ./wdsr_x2 \
    --eval_only &
done