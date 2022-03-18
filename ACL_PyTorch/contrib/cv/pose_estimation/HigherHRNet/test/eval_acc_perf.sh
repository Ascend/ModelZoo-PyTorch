#!/bin/bash

set -eu

datasets_path="./data/coco"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done


echo "====dataset preprocess===="
arch=`uname -m`
rm -rf ./prep_data
rm -rf ./prep_data_flip
python HigherHRNet_preprocess.py --output ./prep_data --output_flip ./prep_data_flip
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python gen_dataset_info.py bin ./prep_data ./prep_bin
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python gen_dataset_info.py bin ./prep_data_flip ./prep_bin_flip
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi


echo "====inference===="
source env.sh
rm -rf result/dumpOutput_device*
python HigherHRNet_benchmark.py --bs 1
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi


echo "====accuracy data===="
python HigherHRNet_postprocess.py  --dump_dir './result/dumpOutput_device0_bs1' --dump_dir_flip './result/dumpOutput_device0_bs1_flip'
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====performance data===="
python HigherHRNet_performance.py
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"