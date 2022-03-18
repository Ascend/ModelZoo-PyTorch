#!/bin/bash
pth_path=$1
onnx_name=$2
om_name=$3
batch_size=$4

if [ $# -ne 4 ]
then 
	echo "parm num is $#, but 4 params is needed"
	echo "format : ./pth2om.sh [pth_path] [onnx_name] [om_name] [batch_size]"
	exit
else 
	echo "pth_path:$1, onnx_name:$2, om_name:$3, batch_size:$4"
fi

dynamic="False"
if [ $batch_size -gt 1 ] 
then
	echo "batch_size is more than 1, onnx model is turned to dynamic"
	dynamic="True"
else
	echo "batch_size is 1"
fi

python ../weights/pth2onnx.py --trained_model=$pth_path --outputName=$onnx_name --dynamic=$dynamic

source ../env.sh

onnx_full_name=$2".onnx"

${install_path}/atc/bin/atc --framework=5 --output=$om_name  --input_format=NCHW  --soc_version=Ascend310 --model=$onnx_full_name --input_shape="input.1:$batch_size,3,550,550"
