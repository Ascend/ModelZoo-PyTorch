#!/bin/bash

datasets_path="/opt/npu"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
# prepare data
if [ ! -d "prep_data" ]
then
    python3.7 sknet_preprocess.py -s ${datasets_path}/imagenet/val -d ./prep_data
    if [ $? != 0 ]; then
        echo "fail to preprocess data!"
        exit -1
    fi
else
    echo "prep_data exists, skip."
fi

# get .info file
if [ ! -f "sknet_prep_bin.info" ]
then
    python3.7 get_info.py bin ./prep_data ./sknet_prep_bin.info 224 224
    if [ $? != 0 ]; then
        echo "fail to get info file!"
        exit -1
    fi
else
    echo "info file exists, skip."
fi

# config environment
source env.sh

# run om
if [ ! -d "result" ]
then 
    ./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=sknet50_1bs.om -input_text_path=./sknet_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
    if [ $? != 0 ]; then
        echo "fail to switch 1bs onnx to om!"
        exit -1
    fi
    ./benchmark.${arch} -model_type=vision -device_id=1 -batch_size=16 -om_path=sknet50_16bs.om -input_text_path=./sknet_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
    if [ $? != 0 ]; then
        echo "fail to switch 16bs onnx to om!"
        exit -1
    fi
else
    echo "om running result exists, skip"
fi

python3.7 vision_metric_ImageNet.py result/dumpOutput_device0/ ${datasets_path}/imagenet/val_label.txt ./ result_1bs.json
if [ $? != 0 ]; then
    echo "fail to calculate accurancy of 1bs om!"
    exit -1
fi
python3.7 vision_metric_ImageNet.py result/dumpOutput_device1/ ${datasets_path}/imagenet/val_label.txt ./ result_16bs.json
if [ $? != 0 ]; then
    echo "fail to calculate accurancy of 16bs om!"
    exit -1
fi
echo "====accuracy data===="
python3.7 test/parse.py result_1bs.json
if [ $? != 0 ]; then
    echo "fail to show accurancy data of 1bs!"
    exit -1
fi
python3.7 test/parse.py result_16bs.json
if [ $? != 0 ]; then
    echo "fail to show accurancy data of 16bs!"
    exit -1
fiss
echo "====performance data===="
python test/parse.py result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail to show performance data of 1bs!"
    exit -1
fi
python3.7 test/parse.py result/perf_vision_batchsize_16_device_1.txt
if [ $? != 0 ]; then
    echo "fail to show performance data of 16bs!"
    exit -1
fi
echo "success"