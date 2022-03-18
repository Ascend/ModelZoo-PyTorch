#!/usr/bin/env bash
rm -rf train_full_1p.log
python3.7.5 -u train.py --num_of_layers 17  --mode S --noiseL 25 --val_noiseL 25 --DeviceID 0 --loss_scale 8 --epochs 70 | tee -a train_full_1p.log
 
