#!/usr/bin/env bash
source ./test/env_npu.sh

eval_file=$(find ./ -name "best_top1_acc_epoch*")
echo "==== Eval top accuracy of epoch pth is ${eval_file}"

python3.7 tools/test.py configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb_1p.py  ${eval_file} --eval top_k_accuracy > ./test/output/0/test_mdoel.log 2>&1

