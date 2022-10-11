#!/bin/bash

datasets_path="/opt/npu"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./prep_dataset
python3.7 imagenet_torch_preprocess.py ${datasets_path}/imagenet/val ./pre_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo '==> 1. creating ./prep_dataset successfully.'

python3.7 get_info.py bin ./pre_dataset/ ./imagenet_prep_bin.info 224 224
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo '==> 2. creating ./imagenet_prep_bin.info successfully.'

source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf ./result/*

./benchmark.${arch} -model_type=vision -batch_size=1 -device_id=2 -om_path=./onnx_alexnet_bs1.om -input_text_path=./imagenet_prep_bin.info -input_width=224 -input_height=224 -useDvpp=false -output_binary=false
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo '==> 3. conducting onnx_alexnet_bs1.om on device 2 successfully.'

./benchmark.${arch} -model_type=vision -batch_size=16 -device_id=3 -om_path=./onnx_alexnet_bs16.om -input_text_path=./imagenet_prep_bin.info -input_width=224 -input_height=224 -useDvpp=false -output_binary=false
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo '==> 4. conducting onnx_alexnet_bs16.om on device 3 successfully.'
python3.7 vision_metric.py --benchmark_out ./result/dumpOutput_device2/ --anno_file ${datasets_path}/imagenet/val_label.txt --result_file ./result/result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo '==> 5. calculate acc on bs1 successfully.'
python3.7 vision_metric.py --benchmark_out ./result/dumpOutput_device3/ --anno_file ${datasets_path}/imagenet/val_label.txt --result_file ./result/result_bs16.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo '==> 6. calculate acc on bs16 successfully.'
echo "====accuracy data===="
python3.7 test/parse.py ./result/result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 test/parse.py ./result/result_bs16.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====performance data===="
python3.7 test/parse.py ./result/perf_vision_batchsize_1_device_2.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 test/parse.py ./result/perf_vision_batchsize_16_device_3.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"