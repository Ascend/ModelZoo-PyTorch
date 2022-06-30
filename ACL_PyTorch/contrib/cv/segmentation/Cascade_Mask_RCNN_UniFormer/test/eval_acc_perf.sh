#!/bin/bash

set -eu

datasets_path="data/coco"
height=800
width=1216

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

echo "==== preprocess ===="
rm -rf val2017_bin
python uniformer_preprocess.py \
    --image_src_path=$datasets_path/val2017/ --bin_file_path=val2017_bin/ \
    --input_height=$height \
    --input_width=$width

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh

echo "==== inference ===="
rm -rf result
./msame --model=uniformer_bs1.om --input=val2017_bin --output=result

echo "==== postprocess ===="
python uniformer_postprocess.py \
    --ann_file_path=$datasets_path/annotations/instances_val2017.json \
    --bin_file_path=result \
    --input_height=$height \
    --input_width=$width

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "success"
