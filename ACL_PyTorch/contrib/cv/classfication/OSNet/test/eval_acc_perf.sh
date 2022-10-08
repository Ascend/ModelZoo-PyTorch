#!/bin/bash

datasets_path="/root/datasets/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./gallery_prep_dataset 
mkdir ./gallery_prep_dataset
python3.7 market1501_torch_preprocess.py /root/datasets/market1501/bounding_box_test ./gallery_prep_dataset
rm -rf ./query_prep_dataset
mkdir ./query_prep_dataset
python3.7 market1501_torch_preprocess.py /root/datasets/market1501/query ./query_prep_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 gen_dataset_info.py bin ./gallery_prep_dataset ./gallery_prep_bin.info 128 256
python3.7 gen_dataset_info.py bin ./query_prep_dataset ./query_prep_bin.info 128 256
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf result/dumpOutput_device0 result/dumpOutput_device1
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=osnet_x1_0_bs1.om -input_text_path=./query_prep_bin.info -input_width=128 -input_height=256 -output_binary=False -useDvpp=False
./benchmark.x86_64 -model_type=vision -device_id=1 -batch_size=1 -om_path=osnet_x1_0_bs1.om -input_text_path=./gallery_prep_bin.info -input_width=128 -input_height=256 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf result/dumpOutput_device2 result/dumpOutput_device3
./benchmark.x86_64 -model_type=vision -device_id=2 -batch_size=16 -om_path=osnet_x1_0_bs16.om -input_text_path=./query_prep_bin.info -input_width=128 -input_height=256 -output_binary=False -useDvpp=False
./benchmark.x86_64 -model_type=vision -device_id=3 -batch_size=16 -om_path=osnet_x1_0_bs16.om -input_text_path=./gallery_prep_bin.info -input_width=128 -input_height=256 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 osnet_x1_0_metrics_market1501.py result/dumpOutput_device0/ result/dumpOutput_device1/ ./ result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 osnet_x1_0_metrics_market1501.py result/dumpOutput_device2/ result/dumpOutput_device3/ ./ result_bs16.json
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
python3.7 test/parse.py result/perf_vision_batchsize_1_device_0.txt result/perf_vision_batchsize_1_device_1.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 test/parse.py result/perf_vision_batchsize_16_device_2.txt result/perf_vision_batchsize_16_device_3.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"
