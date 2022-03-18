#!/bin/bash

datasets_path="/home/datasets/imagenet/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./prep_dataset
python3.7 vit_small_patch16_224_preprocess.py ${datasets_path}/val ./prep_datasets

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source env.sh
rm -rf result/dumpOutput_device0

./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=vit_small_patch16_224_bs1_sim.om -input_text_path=./vit_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False

./benchmark.${arch} -model_type=vision -device_id=1 -batch_size=16 -om_path=vit_small_patch16_224_bs16_sim.om -input_text_path=./vit_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====accuracy data===="
echo "bs1"
python vit_small_patch16_224_postprocess.py result/dumpOutput_device0/ ${datasets_path}/val_label.txt ./ result.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "bs16"
python vit_small_patch16_224_postprocess.py result/dumpOutput_device1/ ${datasets_path}/val_label.txt ./ result.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====performance data===="
python test/parse.py result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python test/parse.py result/perf_vision_batchsize_16_device_1.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"
