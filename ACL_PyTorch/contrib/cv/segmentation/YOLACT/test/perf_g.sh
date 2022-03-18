#!/bin/bash
batch_size=$1
onnx_path=$2

if [ $# -ne 2 ] 
then 
	echo "param num need to be 2, but $# is provided!"
	echo "format : ./perf_g.sh [batch_size] [onnx_path]"
else
	echo "begin trtexec!"
	trtexec --onnx=$onnx_path --fp16 --shapes=input.1:$1x3x550x550
fi

