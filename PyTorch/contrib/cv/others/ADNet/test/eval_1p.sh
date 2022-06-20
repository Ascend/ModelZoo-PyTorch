#!/usr/bin/env bash
rm -rf eval_1p.log
python3.7  test.py --is_distributed 0 --DeviceID 0 --num_gpus 1 --num_of_layers 17 --logdir logssigma25.0_2021-09-25-14-32-10 --test_data BSD68 --test_noiseL 25 | tee -a eval_1p.log
