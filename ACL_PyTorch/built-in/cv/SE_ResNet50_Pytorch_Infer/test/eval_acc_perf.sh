#!/bin/bash

install_path=$1
om=$2
bs=$3
device_id=$4
val_label=$5

source ${install_path}/ascend-toolkit/set_env.sh
chmod +x benchmark.*

echo "eval accuracy"
./benchmark.x86_64 -model_type=vision -om_path=${om} -device_id=${device_id} -batch_size=${bs} -input_text_path=./data/ImageNet_bin.info -input_width=256 -input_height=256 -output_binary=false -useDvpp=false
if [ $? != 0 ]; then
    echo "eval accuracy fail!"
    exit -1
fi

python3 ./vision_metric_ImageNet.py ./result/dumpOutput_device${device_id}/ ${val_label} ./result accuracy_result.json
if [ $? != 0 ]; then
    echo "get accuracy result fail!"
    exit -1
fi
echo "eval accuracy success, result in: ./result/accuracy_result.json"

echo "eval performace"
./benchmark.x86_64 -round=50 -om_path=${om} -device_id=${device_id} -batch_size=${bs}  > ./result/performace_result.json
if [ $? != 0 ]; then
    echo "get accuracy result fail!"
    exit -1
fi
echo "eval performace success, result in: ./result/performace_result.json"

