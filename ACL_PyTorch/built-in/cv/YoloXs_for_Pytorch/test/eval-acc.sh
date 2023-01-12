#!/bin/bash

set -eu

datasets_path="/root/datasets/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./prep_data
python Yolox_preprocess.py --dataroot ${datasets_path} --output './prep_data'

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python gen_dataset_info.py bin ./prep_data ./prep_bin.info 640 640
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf result/



python3 -m ais_bench --model ./models/yolox.om --input prep_data --output ./ --output_dirname result --outfmt BIN


python Yolox_postprocess.py --dataroot ${datasets_path} --dump_dir 'result/'
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "success"

