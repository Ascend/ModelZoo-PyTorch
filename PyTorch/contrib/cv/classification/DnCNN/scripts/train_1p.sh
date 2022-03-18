#!/bin/bash

source scripts/set_npu_env.sh

#python pro_train.py 
python train_1p.py \
  --preprocess True \
  --data_path /data/BSD68  \
  --num_of_layers 17 \
  --mode S \
  --noiseL 15 \
  --val_noiseL 15 \
  --lr 0.01  \
  --epochs 50  \
  --gpu 0  | tee train_1p.log