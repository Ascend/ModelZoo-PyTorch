#!/bin/bash

datasets_path="/opt/npu/cityscapes"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./prep_dataset
python3.7 deeplabv3_torch_preprocess.py ${datasets_path}/leftImg8bit/val ./prep_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python ./gen_dataset_info.py bin ./prep_dataset ./deeplabv3_prep_bin.info 1024 2048
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source env_npu.sh
rm -rf result/dumpOutput_device0
rm -rf result/dumpOutput_device0_bs1
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=./deeplabv3_bs1.om -input_text_path=./deeplabv3_prep_bin.info -input_width=1024 -input_height=2048 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
mv result/dumpOutput_device0 result/dumpOutput_device0_bs1

python3.7 deeplabv3_torch_postprocess.py --output_path=./result/dumpOutput_device0_bs1 --gt_path=${datasets_path}/gtFine/val > result_bs1.json
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

echo "====performance data===="
python3.7 test/parse.py result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "success"