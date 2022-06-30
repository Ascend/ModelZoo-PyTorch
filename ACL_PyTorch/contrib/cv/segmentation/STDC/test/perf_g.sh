#! /bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh

# inference with TensorRT
trtexec --onnx=./stdc_optimize.onnx --shapes=input:1x3x1024x2048 --fp16 --threads --wrokspace=4096