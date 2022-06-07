#!/bin/bash

set -eu
batch_size=64

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
# get quant data
python generate_data.py --img_src_path ${img_src_path} --save_path int8data --batch_size 16
# get quant onnx
python adaptyoloxtiny.py --src_path ${src_path} --save_path ${save_path}

rm -f yolox.om

atc --framework=5 --model=${save_path} --output=yolox  --input_format=NCHW \
--op_precision_mode=op_precision.ini --input_shape="input:$batch_size,3,640,640" --log=error --soc_version=${soc_version}

if [ -f "yolox.om" ]; then
    echo "success"
else
    echo "fail!"
fi
