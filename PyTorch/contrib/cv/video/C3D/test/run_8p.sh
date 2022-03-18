#!/bin/bash
source ./test/env_npu.sh
./tools/dist_train.sh configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py 8 --validate
