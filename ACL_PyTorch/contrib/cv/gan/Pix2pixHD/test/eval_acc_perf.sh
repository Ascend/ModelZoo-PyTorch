#!/bin/bash

arch=`uname -m`
rm -rf ./prep_dataset
python pix2pixhd_preprocess.py ./pix2pixHD/datasets/cityscapes ./prep_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf ./pix2pixhd_prep_bin.info
python get_info.py bin ./prep_dataset ./pix2pixhd_prep_bin.info 2048 1024
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source env.sh
rm -rf result/dumpOutput_device0
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=pix2pixhd_bs1.om -input_text_path=./pix2pixhd_prep_bin.info -input_width=2048 -input_height=1024 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====parse_image===="
echo "bs1"
python pix2pixhd_postprocess.py ./result/dumpOutput_device0 ./generated
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
