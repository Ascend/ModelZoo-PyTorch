#!/bin/bash

datasets_path="/opt/npu"

python3.7 TNT_preprocess.py --src-path=${datasets_path}/imagenet/val --save-path=/home/val_TNT
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 gen_dataset_info.py bin ../val_TNT ./prep_bin.info 16 24
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

source env.sh
rm -rf result/dumpOutput_device0
rm -rf result/dumpOutput_device0_bs1
rm -rf result/dumpOutput_device0_bs16
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=./TNT_bs1.om -input_text_path=./prep_bin.info -input_width=16 -input_height=24 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
mv result/dumpOutput_device0 result/dumpOutput_device0_bs1
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=16 -om_path=./TNT_bs16_sim.om -input_text_path=./prep_bin.info -input_width=16 -input_height=24 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
mv result/dumpOutput_device0 result/dumpOutput_device0_bs16
python3.7 TNT_postprocess.py --label_file=imagenet_labels_tnt.json --pred_dir=./result/dumpOutput_device0_bs1 > result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 TNT_postprocess.py --label_file=imagenet_labels_tnt.json --pred_dir=./result/dumpOutput_device0_bs16 > result_bs16.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="
python3.7 test/parse.py result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 test/parse.py result_bs16.json
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