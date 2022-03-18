#!/bin/bash

datasets_path="./mmaction2-master/data"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./prep_datasets/*
cd mmaction2-master
python ./mmaction/datasets/rawframe_dataset.py ./configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py --output_path /root/c3d/prep_datasets
echo '==> 1. creating ./prep_datasets successfully.'

cd /root/c3d
python get_info.py bin ./prep_datasets ./C3D_prep_bin.info 112 112
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo '==> 2. creating ./C3D_prep_bin.info successfully.'

source env.sh
rm -rf ./result/*

./benchmark.${arch} -model_type=vision -batch_size=1 -device_id=0 -om_path=./C3D.om -input_text_path=./C3D_prep_bin.info -input_width=112 -input_height=112 -useDvpp=false -output_binary=false
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo '==> 3. conducting C3D.om on device 0 successfully.'

python C3D_postprocess.py  ./result/dumpOutput_device0  ${datasets_path}/ucf101/ucf101_val_split_1_rawframes.txt  ./result/top1_acc.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo '==> 4. calculate acc on bs1 successfully.'

echo "====accuracy data===="
python test/parse.py ./result/top1_acc.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====performance data===="
python test/parse.py ./result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "success"
