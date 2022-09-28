#!/bin/bash

datasets_path="/opt/npu/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./prepare_dataset/
mkdir ./prepare_dataset
python3.7 RefineNet_preprocess.py --root-dir ${datasets_path}/VOCdevkit/VOC2012 --bin-dir ./prepare_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 get_info.py bin prepare_dataset ./refinenet_prep_bin.info 500 500
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=model/RefineNet_910_bs1.om -input_text_path=./refinenet_prep_bin.info -input_width=500 -input_height=500 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf result/dumpOutput_device0_bs1
mkdir result/dumpOutput_device0_bs1/
mv result/dumpOutput_device0/* result/dumpOutput_device0_bs1/
cp result/perf_vision_batchsize_1_device_0.txt result/perf_vision_batchsize_1_device_0_bs1.txt
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=16 -om_path=model/RefineNet_910_bs16.om -input_text_path=./refinenet_prep_bin.info -input_width=500 -input_height=500 -output_binary=False -useDvpp=False
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
ulimit -n 10240
python3.7 RefineNet_postprocess.py --val-dir ${datasets_path} --result-dir result/dumpOutput_device0_bs1/
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "bs16"
python3.7 RefineNet_postprocess.py --val-dir ${datasets_path} --result-dir result/dumpOutput_device0_bs16/
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