#!/bin/bash

rm -rf logs
mkdir logs

rm -rf FixRes.onnx
python3.7 FixRes_pth2onnx.py --pretrain_path ResNetFinetune.pth --output_name FixRes.onx
if [ -f "FixRes.onnx" ]; then
    echo "onnx success"
else
    echo "onnx fail!"
fi

rm -rf FixRes_bs1.om FixRes_bs16.om
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# export ASCEND_SLOG_PRINT_TO_STDOUT=1
# export ASCEND_GLOBAL_LOG_LEVEL=1 #debug 0 --> info 1 --> warning 2 --> error 3
# export DUMP_GE_GRAPH=2
atc --framework=5 --model=FixRes.onnx --output=FixRes_bs1 --input_format=NCHW --input_shape="image:1,3,384,384" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"
atc --framework=5 --model=FixRes.onnx --output=FixRes_bs16 --input_format=NCHW --input_shape="image:16,3,384,384" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"

if [ -f "FixRes_bs1.om" ] && [ -f "FixRes_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi
