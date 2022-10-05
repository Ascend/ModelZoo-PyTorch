#!/bin/bash

datasets_path="/root/datasets/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./prep_dataset_query
python3.7 SSD-MobileNetV2_preprocess.py mb2-ssd-lite ${datasets_path}/test/VOC2007/JPEGImages/ ./prep_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 gen_dataset_info.py bin ./prep_dataset ./prep_bin.info 300 300
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf result/dumpOutput_device0
rm -rf result/dumpOutput_device0_bs1
rm -rf result/dumpOutput_device0_bs16
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=./SSD-MobileNetV2_bs1.om -input_text_path=./prep_bin.info -input_width=300 -input_height=300 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
mv result/dumpOutput_device0 result/dumpOutput_device0_bs1
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=16 -om_path=./SSD-MobileNetV2_bs16.om -input_text_path=./prep_bin.info -input_width=300 -input_height=300 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
mv result/dumpOutput_device0 result/dumpOutput_device0_bs16
python3.7 SSD-MobileNetV2_postprocess.py ${datasets_path}/test/VOC2007/ ./voc-model-labels.txt ./result/dumpOutput_device0_bs1 ./eval_result_bs1> result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 SSD-MobileNetV2_postprocess.py ${datasets_path}/test/VOC2007/ ./voc-model-labels.txt ./result/dumpOutput_device0_bs16 ./eval_result_bs16> result_bs16.json
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