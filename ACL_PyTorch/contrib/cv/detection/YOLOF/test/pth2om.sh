#!/bin/bash

set -eu
batch_size=1

for para in $*
do
    if [[ $para == --batch_size* ]]; then
        batch_size=`echo ${para#*=}`
    fi
done


rm -f yolof.onnx

python pytorch2onnx.py \
--model_config YOLOF/playground/detection/coco/yolof/yolof.cspdarknet53.DC5.9x \
--out yolof.onnx \
--pth_path ./YOLOF_CSP_D_53_DC5_9x.pth \
--batch_size $batch_size

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -f yolof.om
source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc --framework=5 --model=yolof.onnx --output=yolof  --input_format=NCHW \
--input_shape="input:$batch_size,3,608,608" --log=error --soc_version=Ascend710

if [ -f "yolof.om" ]; then
    echo "success"
else
    echo "fail!"
fi
