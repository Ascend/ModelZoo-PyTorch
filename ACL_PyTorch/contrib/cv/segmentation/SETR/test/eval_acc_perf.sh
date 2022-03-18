#!/bin/bash

datasets_path=""

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

if [ -n "$datasets_path" ] && [ ! -d "SETR/data" ]; then
    mkdir SETR/data 
    ln -s ${datasets_path}/cityscapes SETR/data
fi

arch=`uname -m`

config_file=configs/SETR/SETR_Naive_768x768_40k_cityscapes_bs_8.py
rm -rf ./input_bin

cd SETR
python3.7  ../SETR_preprocess.py \
        ${config_file} \
        ../input_bin
cd ..
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python3.7 gen_dataset_info.py  bin  ./input_bin setr_768_bin.info 768 768

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

chmod +x ben*
source env.sh
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision \
                    -om_path=setr_naive_768x768_bs1.om \
                    -device_id=0 \
                    -batch_size=1 \
                    -input_text_path=setr_768_bin.info \
                    -input_width=768 -input_height=768 \
                    -useDvpp=False \
                    -output_binary=True
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="
cd SETR
python3.7 ../SETR_postprocess.py ${config_file} ../result/dumpOutput_device0 ../result/merge_output ../miou_eval_result
cd ..
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