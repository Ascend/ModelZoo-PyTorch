#!/bin/bash

data_path=""
arch=`uname -m`

for para in $*
do
    if [[ $para == --data_path* ]]; then
        data_path=`echo ${para#*=}`
    fi
done

python LResNet_preprocess.py 'jpg' ${data_path} './data/lfw';
python LResNet_preprocess.py 'bin' './data/lfw' './lfw.info' 112 112;

source /usr/local/Ascend/ascend-toolkit/set_env.sh;

rm -rf result/dumpOutput_device0;
rm -rf result/dumpOutput_device0_bs1;
rm -rf result/dumpOutput_device0_bs16;
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=./model/model_ir_se100_bs1.om -input_text_path=./lfw.info -input_width=112 -input_height=112 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
mv result/dumpOutput_device0 result/dumpOutput_device0_bs1
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=16 -om_path=./model/model_ir_se100_bs16.om -input_text_path=./lfw.info -input_width=112 -input_height=112 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
mv result/dumpOutput_device0 result/dumpOutput_device0_bs16
python LResNet_postprocess.py ./result/dumpOutput_device0_bs1 ./data/lfw_list.npy > infer_result_bs1.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python LResNet_postprocess.py ./result/dumpOutput_device0_bs16 ./data/lfw_list.npy > infer_result_bs16.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="

bs1_acc=`cat infer_result_bs1.log | grep 'accuracy' | awk '{print $2}'`
echo "bs1_accuracy: ${bs1_acc}"

bs16_acc=`cat infer_result_bs16.log | grep 'accuracy' | awk '{print $2}' `
echo "bs16_accuracy: ${bs16_acc}"

echo "====performance data===="

bs1_one=`cat result/perf_vision_batchsize_1_device_0.txt | grep 'Interface throughputRate' | awk '{print $6}'` 
bs1_fps=`awk 'BEGIN{printf "%.2f\n", 4*'${bs1_one:0:5}'}'`
echo "bs1_fps: ${bs1_fps}"

bs16_one=`cat result/perf_vision_batchsize_16_device_0.txt | grep 'Interface throughputRate' | awk '{print $6}'` 
bs16_fps=`awk 'BEGIN{printf "%.2f\n", 4*'${bs16_one:0:5}'}'`
echo "bs16_fps: ${bs16_fps}"

echo "====success===="