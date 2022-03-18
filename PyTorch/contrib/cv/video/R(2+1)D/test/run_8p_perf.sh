#!/bin/bash
source ./test/env_npu.sh
./tools/dist_train.sh configs/recognition/r2plus1d/r2plus1d_ucf101_rgb_8p_perf.py 8 --validate
