#!/bin/bash

datasets_path="/root/datasets"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./prep_dataset
mkdir prep_dataset
python swin_preprocess.py --cfg=Swin-Transformer/configs/swin_tiny_patch4_window7_224.yaml --data-path=${datasets_path}/imagenet --bin_path=./prep_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1S
fi
python gen_dataset_info.py bin ./prep_dataset ./swin_prep_bin.info 224 224
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=swin_b1.om -input_text_path=./swin_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf result/dumpOutput_device1
./benchmark.${arch} -model_type=vision -device_id=1 -batch_size=16 -om_path=swin_b16.om -input_text_path=./swin_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python swin_postprocess.py --result_path=result/dumpOutput_device0/ --target_file=result/target.json --save_path=./result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python swin_postprocess.py --result_path=result/dumpOutput_device1/ --target_file=result/target.json --save_path=./result_bs16.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====accuracy data===="
python test/parser.py result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python test/parser.py result_bs16.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====performance data===="
python test/parser.py result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python test/parser.py result/perf_vision_batchsize_16_device_1.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"