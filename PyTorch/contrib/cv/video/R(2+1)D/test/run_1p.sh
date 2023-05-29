#!/bin/bash
source ./test/env_npu.sh
python3 ./train.py ./configs/recognition/r2plus1d/r2plus1d_ucf101_rgb_1p.py --validate --seed 0 --deterministic
