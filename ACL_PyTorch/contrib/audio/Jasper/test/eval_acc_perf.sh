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

# genereate result
source env.sh
./benchmark.${arch} -batch_size=1 -om_path=./jasper_1batch.om -round=50 -device_id=0 
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

./benchmark.${arch} -batch_size=16 -om_path=./jasper_16batch.om -round=50 -device_id=0 
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====accuracy data===="

python3.7 om_infer_acl.py \
        --batch_size 1 \
        --model ./jasper_1batch.om \
        --val_manifests ${datasets_path}/librispeech-dev-clean-wav.json \
        --model_config configs/jasper10x5dr_speedp-online_speca.yaml \
        --dataset_dir ${datasets_path} \
        --max_duration 40 \
        --pad_to_max_duration \
        --save_predictions ./result_bs1.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python3.7 om_infer_acl.py \
        --batch_size 16 \
        --model ./jasper_16batch.om \
        --val_manifests ${datasets_path}/librispeech-dev-clean-wav.json \
        --model_config configs/jasper10x5dr_speedp-online_speca.yaml \
        --dataset_dir ${datasets_path} \
        --max_duration 40 \
        --pad_to_max_duration \
        --save_predictions ./result_bs16.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====performance data===="
python3.7 test/parse.py result/PureInfer_perf_of_jasper_1batch_in_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 test/parse.py result/PureInfer_perf_of_jasper_16batch_in_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"
