#!/bin/bash

# onnx
rm -rf hrnet.onnx
python3.7 HRNet_pth2onnx.py --pth=hrnet.pth
# 优化onnx
python3.7 performance_optimize_resize.py hrnet.onnx hrnet.onnx

rm -rf hrnet_bs1.om hrnet_bs4.om
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 将atc日志打印到屏幕
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# 设置日志级别
export ASCEND_GLOBAL_LOG_LEVEL=1 #debug 0 --> info 1 --> warning 2 --> error 3

# 开启ge dump图
# export DUMP_GE_GRAPH=2

# onnx转om模型
# ${chip_name}通过使用"npu-smi info"命令查看

atc --framework=5 --model=hrnet.onnx --output=hrnet_bs1 --input_format=NCHW --input_shape="image:1,3,1024,2048" --log=debug --soc_version=Ascend${chip_name} --out_nodes="Conv_1380:0;Conv_1453:0"

atc --framework=5 --model=hrnet.onnx --output=hrnet_bs4 --input_format=NCHW --input_shape="image:4,3,1024,2048" --log=debug --soc_version=Ascend${chip_name} --out_nodes="Conv_1380:0;Conv_1453:0"

if [ -f "hrnet_bs1.om" ] && [ -f "hrnet_bs4.om" ]; then
    echo "success"
else
    echo "fail!"
fi