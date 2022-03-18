#!/bin/bash

datasets_path="/opt/npu/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

source ./test/env.sh

arch=`uname -m`
rm -rf ./prep_dataset
python3.7 preprocess.py genet ${datasets_path} ./prep_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 get_info.py bin ./prep_dataset ./genet_prep_bin.info 32 32
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=genet_bs1_tuned.om -input_text_path=./genet_prep_bin.info -input_width=32 -input_height=32 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf result/dumpOutput_device1
./benchmark.${arch} -model_type=vision -device_id=1 -batch_size=16 -om_path=genet_bs16_tuned.om -input_text_path=./genet_prep_bin.info -input_width=32 -input_height=32 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 cifar10_acc_eval.py result/dumpOutput_device0/ ./prep_dataset/val_label.txt ./ result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 cifar10_acc_eval.py result/dumpOutput_device0/ ./prep_dataset/val_label.txt ./ result_bs16.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="
python3.7 test/parse.py result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 test/parse.py result_bs16.json
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