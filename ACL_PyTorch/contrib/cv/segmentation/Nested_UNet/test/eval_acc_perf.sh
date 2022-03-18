#!/bin/bash

datasets_path="./dsb2018_96/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./prep_dataset
python3.7 preprocess_nested_unet.py ${datasets_path}/images ./prep_dataset ./val_ids.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 gen_dataset_info.py bin ./prep_dataset ./nested_unet_prep_bin.info 96 96
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source env.sh
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=nested_unet_bs1.om -input_text_path=./nested_unet_prep_bin.info -input_width=96 -input_height=96 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf result/dumpOutput_device1
./benchmark.${arch} -model_type=vision -device_id=1 -batch_size=16 -om_path=nested_unet_bs16.om -input_text_path=./nested_unet_prep_bin.info -input_width=96 -input_height=96 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="
echo "bs1"
python3.7 postprocess_nested_unet.py result/dumpOutput_device0/ ${datasets_path}/masks/0/ 
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "bs16"
python3.7 postprocess_nested_unet.py result/dumpOutput_device1/ ${datasets_path}/masks/0/
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====performance data===="
python3.7 test/parse.py result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 test/parse.py result/perf_vision_batchsize_16_device_1.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"