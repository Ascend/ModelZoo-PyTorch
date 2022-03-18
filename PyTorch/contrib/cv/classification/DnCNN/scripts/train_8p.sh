#!/bin/bash

source scripts/set_npu_env.sh

ulimit -SHn 51200
python train_8p.py \
  --data_path /data/BSD68  \
  --num_of_layers 17 \
  --mode S \
  --noiseL 15 \
  --val_noiseL 15 \
  --epochs 50  \
  --preprocess True \
  --lr 0.008 | tee train_8p.log



