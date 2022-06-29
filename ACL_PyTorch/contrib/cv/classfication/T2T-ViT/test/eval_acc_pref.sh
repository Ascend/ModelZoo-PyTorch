#!/bin/bash

# 配置环境变量
source env.sh

# 数据预处理
python3.7 preprocess.py --data-dir /opt/npu/imagenet/val --out-dir ./imagenet_bs1 --gt-path ./gt_bs1

# msame工具推理
./msame --model "./T2T_ViT_14_bs1_test.om"  --input "./imagenet_bs1/" --output "./out/" --outfmt BIN

#数据后处理并测试精度
python3.7 postprocess.py --result-dir ./out/ --gt-path./gt_bs1.npy

echo "success"