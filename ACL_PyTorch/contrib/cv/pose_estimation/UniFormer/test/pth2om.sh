#!/bin/bash

set -eu
batch_size=1

for para in $*
do
    if [[ $para == --batch_size* ]]; then
        batch_size=`echo ${para#*=}`
    fi
done

rm -f uniformer_dybs.onnx
cd UniFormer/pose_estimation

python tools/pytorch2onnx.py \
    exp/top_down_256x192_global_base/config.py \
    ../../top_down_256x192_global_base.pth \
    --output-file \
    ../../uniformer_dybs.onnx

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

cd ../..
rm -f uniformer_bs$batch_size.om
source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc --framework=5 --model=uniformer_dybs.onnx --output=uniformer_bs$batch_size --input_format=NCHW \
    --input_shape="input:$batch_size,3,256,192" --log=error --soc_version=Ascend710

if [ -f "uniformer_bs$batch_size.om" ]; then
    echo "success"
else
    echo "fail!"
fi
