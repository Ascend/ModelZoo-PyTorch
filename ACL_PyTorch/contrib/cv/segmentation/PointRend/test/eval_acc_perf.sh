#!/bin/bash

source env.sh

datasets_path="./cityscapes"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

echo "===========  start preprocess  =============="
echo "img shape = 3*2048*1024"
rm -rf ./preprocess_bin
python3.7 PointRend_preprocess.py ${datasets_path} ./preprocess_bin
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "get_dateset_info start!"
rm -rf ./prep_bin.info
python3.7 gen_dataset_info.py bin ./preprocess_bin ./prep_bin.info 2048 1024
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "===========  start benchmark  =============="
echo "benchmark batch1"
rm -rf result/dumpOutput_device0
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./PointRend_bs1.om -input_text_path=./prep_bin.info -input_width=2048 -input_height=1024 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "===========  start eval acc  =============="
echo 'eval acc batch_size = 1'
python3.7 PointRend_postprocess.py ${datasets_path} ./result/dumpOutput_device0/
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
sleep 5
echo "acc result  baseline mIOU:78.86"

echo "===========  start eval perf  =============="
echo 'eval perf batch_size = 1'
python3.7 test/parse.py result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"

