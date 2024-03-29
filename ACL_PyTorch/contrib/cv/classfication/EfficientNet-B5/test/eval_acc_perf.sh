#!/bin/bash

datasets_path="/home/common/datasets/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./prep_dataset
python3 imagenet_torch_preprocess.py efficientnetB5 ${datasets_path}/imagenet/val ./prep_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3 get_info.py bin ./prep_dataset ./efficientnetb5.info 456 456
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=efficientnetb5_bs1.om -input_text_path=./efficientnetb5.info -input_width=456 -input_height=456 -output_binary=False -useDvpp=False

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf result/dumpOutput_device1
./benchmark.${arch} -model_type=vision -device_id=1 -batch_size=16 -om_path=efficientnetb5_bs16.om -input_text_path=./efficientnetb5.info -input_width=456 -input_height=456 -output_binary=False -useDvpp=False

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python3 vision_metric_ImageNet.py result/dumpOutput_device0/ ${datasets_path}/imagenet/val_label.txt ./ result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3 vision_metric_ImageNet.py result/dumpOutput_device1/ ${datasets_path}/imagenet/val_label.txt ./ result_bs16.json
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
