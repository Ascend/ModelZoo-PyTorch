#!/usr/bin/env bash
rm -rf train_full_8p.log
nohup python3.7 -m torch.distributed.launch --nproc_per_node=8 train.py --is_distributed 1 --DeviceID 0,1,2,3,4,5,6,7 --num_gpus 8 --world_size 8 --loss_scale 8 --num_of_layers 17  --mode S --noiseL 25 --val_noiseL 25 --epochs 70 --lr 1e-3 --batchSize 128 | tee -a train_full_8p.log
