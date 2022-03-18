#!/bin/bash

source test/env.sh

nohup python3.7 -u train.py --size 736 --apex --pth_path saved_model/3T736_latest.pth --epoch_num 3 --val_interval 1