#!/bin/sh
currentDir=$(cd "$(dirname "$0")"; pwd)

if [ ! -d ${currentDir}/ckpt ]; then
  mkdir ${currentDir}/ckpt
fi

source ${currentDir}/npu_set_env.sh

python3.7 ${currentDir}/train.py \
          --data_dir=${currentDir}/data/ \
          --pretrained_dir=${currentDir}/ViT-B_16.npz \
          --addr=10.155.170.67 \
          --name=ViTBase \
          --train_batch_size=64 \
          --num_steps=10000 \
          --fp16 \
          --npu-fused-sgd \
          --combine-grad \
          --output_dir=${currentDir}/ckpt > vitbase.log 2>&1 
