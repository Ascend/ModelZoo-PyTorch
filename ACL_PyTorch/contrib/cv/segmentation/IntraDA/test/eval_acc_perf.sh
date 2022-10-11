#!/bin/bash

datasets_path="/opt/npu/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./val2017_bin
python3.7 intrada_preprocess.py ${datasets_path}cityscapes/ ./pre_dataset_bin
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 gen_dataset_info.py bin ./pre_dataset_bin ./intraDA_deeplabv2_pre_bin_512_1024.info 512 1024
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
chmod +x ben*
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf result/dumpOutput_device0
rm -rf result/dumpOutput_device1
./benchmark.${arch}  -model_type=vision -device_id=0 -batch_size=1 -om_path=./intraDA_deeplabv2_bs1.om -input_text_path=./intraDA_deeplabv2_pre_bin_512_1024.info -input_width=1024 -input_height=512 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
./benchmark.${arch}  -model_type=vision -device_id=1 -batch_size=16 -om_path=./intraDA_deeplabv2_bs16.om -input_text_path=./intraDA_deeplabv2_pre_bin_512_1024.info -input_width=1024 -input_height=512 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====accuracy data===="
python3.7 -u intrada_postprocess.py ${datasets_path}cityscapes ./result/dumpOutput_device0 ./out
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 -u intrada_postprocess.py ${datasets_path}cityscapes ./result/dumpOutput_device1 ./out
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