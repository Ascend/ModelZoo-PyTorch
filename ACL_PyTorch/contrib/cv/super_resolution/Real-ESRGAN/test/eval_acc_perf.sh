#!/bin/bash

datasets_path="./Real-ESRGAN"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch="x86_64"
rm -rf ./prep_dataset_bin
python Real-ESRGAN_preprocess.py ${datasets_path}/inputs ./prep_dataset_bin 220 220
if [ $? != 0 ]; then
    echo "preprocess fail!"
fi
rm -rf ./prep_bin.info
python gen_dataset_info.py bin ./prep_dataset_bin ./prep_bin.info 220 220
if [ $? != 0 ]; then
    echo "get_dataset_info fail!"
fi
source env.sh
rm -rf result/dumpOutput_device0_bs1
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=./realesrgan_bs1-220.om -input_text_path=./prep_bin.info -input_width=220 -input_height=220 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "220 size benchmark is fail!"
fi
./benchmark.${arch} -round=20 -om_path=realesrgan_bs1.om -device_id=0 -batch_size=1
./benchmark.${arch} -round=20 -om_path=realesrgan_bs16.om -device_id=0 -batch_size=16
rm -rf ./img_path
python Real-ESRGAN_postprocess.py  ./result/dumpOutput_device0 ./img_path
if [ $? != 0 ]; then
    echo "postprocess is fail!"
fi

echo "====performance data===="
python test/parse.py  ./result/PureInfer_perf_of_realesrgan_bs1_in_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
fi
python test/parse.py  ./result/PureInfer_perf_of_realesrgan_bs16_in_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
fi
echo "success"
