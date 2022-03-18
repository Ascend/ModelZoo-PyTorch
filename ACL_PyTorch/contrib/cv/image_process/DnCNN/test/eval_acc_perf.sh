#!/bin/bash

datasets_path="./data"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`

rm -rf ./ISource
rm -rf ./INoisy
python3.7 data_preprocess.py ./data ISource INoisy
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 get_info.py bin INoisy DnCNN_bin.info 481 481
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source env.sh
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=DnCNN-S-15_bs1.om -input_text_path=./DnCNN_bin.info -input_width=481 -input_height=481 -output_binary=true -useDvpp=false
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf result/dumpOutput_device1
./benchmark.${arch} -model_type=vision -device_id=1 -batch_size=16 -om_path=DnCNN-S-15_bs16.om -input_text_path=./DnCNN_bin.info -input_width=481 -input_height=481 -output_binary=true -useDvpp=false
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf result_bs1.log
python3.7 postprocess.py result/dumpOutput_device0 > result_bs1.log 
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf result_bs16.log
python3.7 postprocess.py result/dumpOutput_device1 > result_bs16.log 
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo -e "\n"
echo "==== accuracy data ===="

echo "==== 310 bs1 PSNR ===="
python3.7 test/parse.py result_bs1.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "==== 310 bs16 PSNR ===="
python3.7 test/parse.py result_bs16.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo -e "\n"
echo "==== performance data ===="
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
echo -e "\n"
echo "success"
