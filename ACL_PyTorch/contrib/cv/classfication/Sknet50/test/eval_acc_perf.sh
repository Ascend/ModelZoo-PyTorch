#!/bin/bash

datasets_path="/root/datasets/"
for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`

echo "====data preprocess===="
rm -rf ./prep_data
python sknet_preprocess.py -s ${datasets_path}/imageNet/val -d ./prep_data
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf sknet_prep_bin_lsl.info
python get_info.py bin ./prep_data ./sknet_prep_bin_lsl.info 224 224
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

chmod +x benchmark.x86_64
chmod +x benchmark.aarch64
source /usr/local/Ascend/ascend-toolkit/set_env.sh
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "bs1"
echo "====performance data===="
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=sk_resnet50_bs1_310P.om -input_text_path=sknet_prep_bin_lsl.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="
rm -rf result_bs1.json
python vision_metric_ImageNet.py result/dumpOutput_device0/ ${datasets_path}/imageNet/val_label_LSL.txt ./ result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "bs8"
echo "====performance data===="
rm -rf result/dumpOutput_device1
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=8 -om_path=sk_resnet50_bs8_310P.om -input_text_path=sknet_prep_bin_lsl.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="
rm -rf result_bs8.json
python vision_metric_ImageNet.py result/dumpOutput_device0/ ${datasets_path}/imageNet/val_label_LSL.txt ./ result_bs8.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====performance data===="
python test/parse.py result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python test/parse.py result/perf_vision_batchsize_8_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"