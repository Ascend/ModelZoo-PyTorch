#!/bin/bash

source env.sh

arch=`uname -m`
# generate prep_dataset
rm -rf ./prep_dataset/
python3.7 dcgan_preprocess.py  ./prep_dataset/
if [ $? != 0 ]; then
    echo "preprocess fail!"
    exit -1
fi
# generate dataset info
rm -rf ./kinetics.info
python3.7 get_info.py bin ./data/kinetics-skeleton/val_data ./kinetics.info 300 18
if [ $? != 0 ]; then
    echo "info fail!"
    exit -1
fi
# benchmark bs1
rm -rf ./result/dumpOutput_device0/ perf_vision_batchsize_1_device_0.txt
./benchmark.${arch} -model_type=vision -batch_size=1 -device_id=0 -om_path=./st-gcn_bs1.om  -input_width=300 -input_height=8 -input_text_path=./kinetics.info -useDvpp=false -output_binary=true
if [ $? != 0 ]; then
    echo "benchmark bs1 fail!"
    exit -1
fi
# benchmark bs16
rm -rf ./result/dumpOutput_device1/ perf_vision_batchsize_16_device_1.txt
./benchmark.${arch} -model_type=vision -batch_size=1 -device_id=1 -om_path=./st-gcn_bs1.om  -input_width=300 -input_height=8 -input_text_path=./kinetics.info -useDvpp=false -output_binary=true
if [ $? != 0 ]; then
    echo "benchmark bs16 fail!"
    exit -1
fi
# print performance data
echo "====performance data===="
python3.7 test/parse.py result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "parse bs1 fail!"
    exit -1
fi
python3.7 test/parse.py result/perf_vision_batchsize_16_device_1.txt
if [ $? != 0 ]; then
    echo "parse bs16 fail!"
    exit -1
fi
echo "success"