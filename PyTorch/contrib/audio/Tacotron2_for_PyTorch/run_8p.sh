#!/usr/bin/env bash
source env_npu.sh
mkdir -p output

export HCCL_CONNECT_TIMEOUT=1800
export RANK_SIZE=8
KERNEL_NUM=$(($(nproc)/8))
currentDir=$(cd "$(dirname "$0")";pwd)

# 训练epoch 301
train_epochs=301

# 接收传入参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --train_epochs* ]];then
        train_epochs=`echo ${para#*=}`
    fi
done

for((RANK_ID=0;RANK_ID<RANK_SIZE;RANK_ID++))
do
    if [ $(uname -m) = "aarch64" ]
    then
        PID_START=$((KERNEL_NUM * RANK_ID))
        PID_END=$((PID_START + KERNEL_NUM - 1))
        taskset -c $PID_START-$PID_END python3.7 -u ${currentDir}/train.py \
            -m Tacotron2 \
            -o ./output/ \
            --amp \
            -lr 1e-3 \
            --epochs ${train_epochs} \
            -bs 128 \
            --weight-decay 1e-6 \
            --grad-clip-thresh 1.0 \
            --cudnn-enabled \
            --load-mel-from-disk \
            --training-files=filelists/ljs_mel_text_train_filelist.txt \
            --validation-files=filelists/ljs_mel_text_val_filelist.txt \
            --log-file nvlog.json \
            --anneal-steps 500 1000 1500 \
            --anneal-factor 0.3 \
            --rank ${RANK_ID} \
            --seed 0 \
            --dist-backend 'hccl' > npu8p.log 2>&1 &
    else
        python3.7 -u ${currentDir}/train.py \
            -m Tacotron2 \
            -o ./output/ \
            --amp \
            -lr 1e-3 \
            --epochs 301 \
            -bs 128 \
            --weight-decay 1e-6 \
            --grad-clip-thresh 1.0 \
            --cudnn-enabled \
            --load-mel-from-disk \
            --training-files=filelists/ljs_mel_text_train_filelist.txt \
            --validation-files=filelists/ljs_mel_text_val_filelist.txt \
            --log-file nvlog.json \
            --anneal-steps 500 1000 1500 \
            --anneal-factor 0.3 \
            --rank ${RANK_ID} \
            --seed 0 \
            --dist-backend 'hccl' > npu8p.log 2>&1 &
    fi
done