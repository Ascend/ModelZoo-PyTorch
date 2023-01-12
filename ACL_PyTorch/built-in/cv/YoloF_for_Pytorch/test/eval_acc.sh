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
python3 YOLOF_preprocess.py --image_src_path ${datasets_path}/val2017

python3 gen_dataset_info.py \
${datasets_path} \
${mmdetection_path}/configs/yolof/yolof_r50_c5_8x8_1x_coco.py  \
val2017_bin  val2017_bin_meta  \
yolof.info  yolof_meta.info  \
640 640

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi


rm -rf result

python3 -m ais_bench --model yolof.om --input val2017_bin --output ./ --output_dirname result --outfmt BIN
rm -rf results.txt
python3 YOLOF_postprocess.py --dataset_path ${datasets_path} --model_config ${mmdetection_path}/configs/yolof/yolof_r50_c5_8x8_1x_coco.py
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "success"

