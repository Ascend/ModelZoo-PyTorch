#!/bin/bash

set -eu

datasets_path="/root/datasets"
batch_size=1
mmdetection_path="/home"
for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
    if [[ $para == --batch_size* ]]; then
        batch_size=`echo ${para#*=}`
    fi
    if [[ $para == --mmdetection_path* ]]; then
        mmdetection_path=`echo ${para#*=}`
    fi
done


arch=`uname -m`

rm -rf val2017_bin
rm -rf val2017_bin_meta
python YOLOX_preprocess.py --image_src_path ${datasets_path}/val2017

python gen_dataset_info.py \
${datasets_path} \
${mmdetection_path}/configs/yolox/yolox_s_8x8_300e_coco.py  \
val2017_bin  val2017_bin_meta  \
yolox.info  yolox_meta.info  \
640 640

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi


rm -rf result

./benchmark.${arch} -model_type=vision -om_path=yolox.om -device_id=0 -batch_size=${batch_size} \
-input_text_path=yolox.info -input_width=640 -input_height=640 -useDvpp=false -output_binary=true

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf result.txt
echo "====performance data===="
python test/parse.py result/perf_vision_batchsize_${batch_size}_device_0.txt

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python YOLOX_postprocess.py --dataset_path ${datasets_path} --model_config ${mmdetection_path}/configs/yolox/yolox_tiny_8x8_300e_coco.py
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi



echo "success"

