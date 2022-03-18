#!/bin/bash

datasets_path="/opt/npu/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

# 数据预处理
echo "====dataset preprocess===="
arch=`uname -m`
rm -rf ./prep_dataset
mkdir prep_dataset
python3.7 HRNet_preprocess.py --src_path=${datasets_path} --save_path=./prep_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 gen_dataset_info.py bin ./prep_dataset ./hrnet_prep_bin.info 2048 1024
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

# 离线推理
echo "====inference===="
source env.sh
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=hrnet_bs1.om -input_text_path=./hrnet_prep_bin.info -input_width=2048 -input_height=1024 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf result/dumpOutput_device1
./benchmark.${arch} -model_type=vision -device_id=1 -batch_size=4 -om_path=hrnet_bs4.om -input_text_path=./hrnet_prep_bin.info -input_width=2048 -input_height=1024 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====accuracy data===="
echo "bs1"
python3.7 HRNet_postprocess.py result/dumpOutput_device0/ ${datasets_path}/cityscapes/gtFine/val/ ./ result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "bs4"
python3.7 HRNet_postprocess.py result/dumpOutput_device1/ ${datasets_path}/cityscapes/gtFine/val/ ./ result_bs4.json
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

python3.7 test/parse.py result/perf_vision_batchsize_4_device_1.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "success"