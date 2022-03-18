#!/usr/bin/env bash
source ./test/env_npu.sh
python3.7 tools/test.py configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py     work_dirs/npu_8p/latest.pth --eval top_k_accuracy

