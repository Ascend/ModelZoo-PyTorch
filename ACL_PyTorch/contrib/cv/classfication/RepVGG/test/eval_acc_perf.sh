#!/bin/bash

datasets_path="/root/datasets/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./prep_dataset
python RepVGG_preprocess.py repvgg ${datasets_path}/imagenet/val ./prep_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm RepVGG_prep_bin.info
python gen_dataset_info.py bin ./prep_dataset ./RepVGG_prep_bin.info 224 224
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source env.sh
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=RepVGG_bs1.om -input_text_path=./RepVGG_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf result/dumpOutput_device1
./benchmark.${arch} -model_type=vision -device_id=1 -batch_size=16 -om_path=RepVGG_bs16.om -input_text_path=./RepVGG_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="
echo "bs1"
python RepVGG_postprocess.py result/dumpOutput_device0/ ${datasets_path}/imagenet/val_label.txt ./ result.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "bs16"
python RepVGG_postprocess.py result/dumpOutput_device1/ ${datasets_path}/imagenet/val_label.txt ./ result.json
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