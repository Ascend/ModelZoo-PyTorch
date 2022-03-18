#!/usr/bin/env bash
source env_npu.sh

export RANK_SIZE=8
rm -f nohup.out

KERNEL_NUM=$(($(nproc)/8))
for((RANK_ID=0;RANK_ID<RANK_SIZE;RANK_ID++))
do
    export RANK_ID=$RANK_ID

    if [ $(uname -m) = "aarch64" ]
    then
        PID_START=$((KERNEL_NUM * RANK_ID))
        PID_END=$((PID_START + KERNEL_NUM - 1))
        taskset -c $PID_START-$PID_END python3.7 fine_tune_new_8p.py \
            --seed 12345 \
            --amp_cfg \
            --opt_level O2 \
            --loss_scale_value 1024 \
            --device_list '0,1,2,3,4,5,6,7' \
            --batch_size 4096 \
            --epochs 8 \
            --epochs_per_save 1 \
            --lr 0.005 \
            --workers 64 \
            --data_dir '/home/VGG-Face2/data/train_cropped' \
            --addr=$(hostname -I |awk '{print $1}') \
            --rank 0 \
            --dist_url 'tcp://127.0.0.1:50000' \
            --dist_backend 'hccl' \
            --multiprocessing_distributed \
            --world_size 1 &
    else
        python3.7 fine_tune_new_8p.py \
            --seed 12345 \
            --amp_cfg \
            --opt_level O2 \
            --loss_scale_value 1024 \
            --device_list '0,1,2,3,4,5,6,7' \
            --batch_size 4096 \
            --epochs 8 \
            --epochs_per_save 1 \
            --lr 0.005 \
            --workers 64 \
            --data_dir '/home/VGG-Face2/data/train_cropped' \
            --addr=$(hostname -I |awk '{print $1}') \
            --rank 0 \
            --dist_url 'tcp://127.0.0.1:50000' \
            --dist_backend 'hccl' \
            --multiprocessing_distributed \
            --world_size 1 &
    fi
done
wait


