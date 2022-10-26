#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh

arch=`uname -m`
# generate prep_dataset
rm -rf ./prep_dataset/
python3.7 dcgan_preprocess.py  ./prep_dataset/
if [ $? != 0 ]; then
    echo "preprocess fail!"
    exit -1
fi
# generate dataset info
rm -rf ./dcgan_prep_bin.info
python3.7 get_info.py bin ./prep_dataset/ ./dcgan_prep_bin.info 1 1
if [ $? != 0 ]; then
    echo "info fail!"
    exit -1
fi
# benchmark bs1
rm -rf ./result/dumpOutput_device0/ perf_vision_batchsize_1_device_0.txt
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=./dcgan_sim_bs1.om -input_text_path=./dcgan_prep_bin.info -input_width=1 -input_height=1 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "benchmark bs1 fail!"
    exit -1
fi
# benchmark bs16
rm -rf ./result/dumpOutput_device1/ perf_vision_batchsize_16_device_1.txt
./benchmark.${arch} -model_type=vision -device_id=1 -batch_size=16 -om_path=./dcgan_sim_bs16.om -input_text_path=./dcgan_prep_bin.info -input_width=1 -input_height=1 -output_binary=True -useDvpp=False
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