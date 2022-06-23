#!/bin/bash


set -eu

mkdir -p logs

h_list=(240 480 720 1080)
w_list=(320 640 1280 1920)
length=${#h_list[@]}
workspace=50000

trtexec  --onnx=EDSR_x2.onnx --minShapes="image:1x3x240x320" --optShapes="image:1x3x100x100" --maxShapes="image:1x3x1080x1920" --saveEngine=EDSR_x2.trt --fp16 --threads --workspace=$workspace

for ((i=0; i<${length}; i++));
do
    h=${h_list[$i]}
    w=${w_list[$i]}
    trtexec --loadEngine=EDSR_x2.trt --threads --workspace=${workspace} --shapes="image:1x3x${h}x${w}"
done

