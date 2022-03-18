#!/bin/bash
source ./test/env_npu.sh
python3.7 ./train.py ./configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb_1p.py --validate --seed 0 --deterministic
