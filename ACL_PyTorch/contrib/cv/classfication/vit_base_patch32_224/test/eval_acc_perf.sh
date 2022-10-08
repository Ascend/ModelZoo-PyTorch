#!/bin/bash

datasets_path="/root/datasets/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./prep_dataset
python3.7 vit_base_patch32_224_preprocess.py --data-path ${datasets_path}/imagenet/val --store-path ./prep_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 gen_dataset_info.py bin prep_dataset ./vit_prep_bin.info 224 224
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=vit_bs1.om -input_text_path=./vit_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python3.7 vit_base_patch32_224_preprocess.py


echo "====accuracy data for bs1===="
python3.7 vit_base_patch32_224_postprocess.py --output result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====performance data for bs1===="
./benchmark.x86_64 -round=50 -om_path=vit_bs1.om -device_id=0 -batch_size=1
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi


rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=16 -om_path=vit_bs16.om -input_text_path=./vit_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi


echo "====accuracy data for bs16===="
python3.7 vit_base_patch32_224_postprocess.py --output result_bs16.json

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====performance data for bs16===="
./benchmark.x86_64 -round=50 -om_path=vit_bs16.om -device_id=0 -batch_size=16
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi


echo "success"