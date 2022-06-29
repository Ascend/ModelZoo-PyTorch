#!/bin/bash

set -eu
batch_size=1

for para in $*
do
    if [[ $para == --batch_size* ]]; then
        batch_size=`echo ${para#*=}`
    fi
    if [[ $para == --soc_version* ]]; then
        soc_version=`echo ${para#*=}`
    fi
done


cd mmdetection

python tools/deployment/pytorch2onnx.py \
configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
../mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth \
--output-file ../swin_net_bs$batch_size.onnx \
--batch_size=$batch_size

python ../swin_mod_newroi.py $batch_size

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

cd ..
source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc --framework=5 --model=swin_mod_bs$batch_size.onnx --output=swin  --input_format=NCHW \
--log=error --soc_version=${soc_version} --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance

if [ -f "swin.om" ]; then
    echo "success"
else
    echo "fail!"
fi
