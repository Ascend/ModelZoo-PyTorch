#!/bin/bash

rm -rf tnt_s_patch16_224_bs1.onnx tnt_s_patch16_224_bs16.onnx tnt_s_patch16_224_bs1_cast.onnx tnt_s_patch16_224_bs16_cast.onnx
python3.7 TNT_pth2onnx.py --pretrain_path tnt_s_81.5.pth.tar --batch_size 1 > logs/onnx_bs1.log
python3.7 TNT_pth2onnx.py --pretrain_path tnt_s_81.5.pth.tar --batch_size 16 > logs/onnx_bs16.log
python3.7 -m onnxsim tnt_s_patch16_224_bs16_cast.onnx tnt_s_patch16_224_bs16_cast_sim.onnx --input-shape "16,196,16,24"
if [ -f "tnt_s_patch16_224_bs1.onnx" ] && [ -f "tnt_s_patch16_224_bs16.onnx" ]; then
    echo "onnx success"
else
    echo "onnx fail!"
fi

if [ -f "tnt_s_patch16_224_bs1_cast.onnx" ] && [ -f "tnt_s_patch16_224_bs16_cast.onnx" ]; then
    echo "cast success"
else
    echo "cast fail"
    exit -1
fi

rm -rf TNT_bs1.om TNT_bs16.om
source env.sh
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=1 #debug 0 --> info 1 --> warning 2 --> error 3
# export DUMP_GE_GRAPH=2
atc --framework=5 --model=tnt_s_patch16_224_bs1_cast.onnx --output=TNT_bs1 --input_format=NCHW --input_shape="inner_tokens:1,196,16,24" --log=debug --soc_version=Ascend310 > logs/atc_bs1.log
atc --framework=5 --model=tnt_s_patch16_224_bs16_cast_sim.onnx --output=TNT_bs16_sim --input_format=NCHW --input_shape="inner_tokens:16,196,16,24" --log=debug --soc_version=Ascend310 > logs/atc_bs16.log

if [ -f "TNT_bs1.om" ] && [ -f "TNT_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi