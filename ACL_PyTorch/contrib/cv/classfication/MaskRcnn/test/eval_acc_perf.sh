#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh

datasets_path="/root/datasets/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./prep_dataset
mkdir -p prep_dataset
python3.7 imagenet_torch_preprocess.py ${datasets_path}/imageNet/val ./prep_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 get_info.py bin ./prep_dataset ./dataset_prep_bin.info 224 224
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -batch_size=1 -device_id=0 -input_text_path=./dataset_prep_bin.info -input_width=224 -input_height=224 -om_path=./moco-v2-atc-bs1.om -useDvpp=False -output_binary=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf result/dumpOutput_device1
./benchmark.${arch} -model_type=vision -batch_size=16 -device_id=1 -input_text_path=./dataset_prep_bin.info -input_width=224 -input_height=224 -om_path=./moco-v2-atc-bs16.om -useDvpp=False -output_binary=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python3.7 vision_metric_ImageNet.py result/dumpOutput_device0/ ${datasets_path}/imageNet/val_label.txt . result_bs1.json
python3.7 vision_metric_ImageNet.py result/dumpOutput_device1/ ${datasets_path}/imageNet/val_label.txt . result_bs16.json

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
