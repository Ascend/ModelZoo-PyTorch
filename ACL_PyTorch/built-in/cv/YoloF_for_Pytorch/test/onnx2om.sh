#!/bin/bash

set -eu
batch_size=16

for para in $*
do
    if [[ $para == --batch_size* ]]; then
        batch_size=`echo ${para#*=}`
    fi
done

img_src_path=$1
src_path=$2
save_path=$3
soc_version=$4
if [ ! -d "int8data" ];then
    mkdir int8data
fi
# get quant data
python generate_data.py --img_src_path ${img_src_path} --save_path int8data --batch_size 16
# get quant onnx
python adaptyolof.py --src_path ${src_path} --save_path ${save_path}

rm -f yolof.om

atc --framework=5 --model=${save_path} --output=yolof  --input_format=NCHW \
--op_precision_mode=op_precision.ini --input_shape="input:$batch_size,3,640,640" --log=error --soc_version=${soc_version}

if [ -f "yolof.om" ]; then
    echo "success"
else
    echo "fail!"
fi
