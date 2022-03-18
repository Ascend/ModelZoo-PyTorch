#!/bin/bash

datasets_path="/root/.torch/datasets/citys"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

rm -rf ./prepare_dataset/
python3.7 ENet_preprocess.py --src-path=$datasets_path --save_path ./prep_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python3.7 get_info.py bin ./prep_dataset ./enet_prep_bin.info 480 480
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

source env.sh
rm -rf result/dumpOutput_device0
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=models/enet_citys_910_bs1.om -input_text_path=./enet_prep_bin.info -input_width=480 -input_height=480 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf result/dumpOutput_device0_bs1
mkdir result/dumpOutput_device0_bs1/
mv result/dumpOutput_device0/* result/dumpOutput_device0_bs1/
cp result/perf_vision_batchsize_1_device_0.txt result/perf_vision_batchsize_1_device_0_bs1.txt
rm -rf result/dumpOutput_device0

./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=models/enet_citys_910_bs16.om -input_text_path=./enet_prep_bin.info -input_width=480 -input_height=480 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf result/dumpOutput_device0_bs16
mkdir result/dumpOutput_device0_bs16/
mv result/dumpOutput_device0/* result/dumpOutput_device0_bs16/
cp result/perf_vision_batchsize_16_device_0.txt result/perf_vision_batchsize_16_device_0_bs16.txt

echo "====accuracy data===="
echo "bs1"
python3.7 ENet_postprocess.py --src-path=$datasets_path  --result-dir result/dumpOutput_device0_bs1/
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "bs16"
python3.7 ENet_postprocess.py --src-path=$datasets_path  --result-dir result/dumpOutput_device0_bs16/
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
python3.7 test/parse.py result/perf_vision_batchsize_16_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"

