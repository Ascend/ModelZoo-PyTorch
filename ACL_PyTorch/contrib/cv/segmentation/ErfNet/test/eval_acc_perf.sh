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
python ErfNet_preprocess.py ${datasets_path}/cityscapes/leftImg8bit/val ./prep_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python gen_dataset_info.py bin ./prep_dataset ./ErfNet_prep_bin.info 1024 512
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=ErfNet_bs1.om -input_text_path=./ErfNet_prep_bin.info -input_width=1024 -input_height=512 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf result/dumpOutput_device1
./benchmark.${arch} -model_type=vision -device_id=1 -batch_size=16 -om_path=ErfNet_bs16.om -input_text_path=./ErfNet_prep_bin.info -input_width=1024 -input_height=512 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="
echo "bs1"
python ErfNet_postprocess.py result/dumpOutput_device0/ ${datasets_path}/cityscapes/gtFine/val/ 
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "bs16"
python ErfNet_postprocess.py result/dumpOutput_device1/ ${datasets_path}/cityscapes/gtFine/val/
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