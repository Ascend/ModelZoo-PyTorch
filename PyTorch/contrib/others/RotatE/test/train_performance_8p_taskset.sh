#!/usr/bin/env bash
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export RANK_SIZE=8
rm -f nohup.out


lr="0.00005"
bs="1024"
filename="train_perf_8p_task"
filename="test8p"
index=0
for para in $*
do
    if [[ $para == --lr* ]]; then
        lr=`echo ${para#*=}`
    fi
    if [[ $para == --bs* ]]; then
        bs=`echo ${para#*=}`
    fi
    if [[ $para == --filename* ]]; then
        filename=`echo ${para#*=}`
    fi
    if [[ $para == --f* ]]; then
        filename=`echo ${para#*=}`
    fi
    if [[ $para == --index* ]]; then
        index=`echo ${para#*=}`
    fi
    if [[ $para == --i* ]]; then
        index=`echo ${para#*=}`
    fi
done
for((RANK_ID=0;RANK_ID<RANK_SIZE;RANK_ID++))
do
    export RANK_ID=$RANK_ID
    let a=0+RANK_ID*24
    let b=23+RANK_ID*24
    nohup taskset -c $a-$b python3 codes/apex_run_mp.py \
        --do_train \
        --npu \
        --data_path data/FB15k-237 \
        --model RotatE \
        -n 256 \
        -b ${bs} \
        -d 1000 \
        -g 9.0 \
        -a 1.0 \
        -adv \
        -lr ${lr} \
        --max_steps 1000 \
        --warm_up_steps 150000 \
        -save models/${filename} \
        --test_batch_size 16 \
        -de \
        --apex \
        --apex_level O0 \
        --loss_scale 1.0 \
        --distributed \
        --world_size 8 \
        --port 29688 \
        -cpu 8 \
        --test_cuda \
        --backend hccl &
done