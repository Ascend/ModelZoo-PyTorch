#!/bin/bash
PTH_FILE=$1

source ./test/env_npu.sh

python3.7 pth2onnx.py --input_file $PTH_FILE --output_file './StarGAN.onnx'
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest

atc --framework=5 --model=StarGAN.onnx --output=StarGAN_bs1 --input_format=NCHW \
    --input_shape="real_img:1,3,128,128;attr:1,5" --log=debug --soc_version=Ascend310 

atc --framework=5 --model=StarGAN.onnx --output=StarGAN_bs4 --input_format=NCHW \
    --input_shape="real_img:4,3,128,128;attr:4,5" --log=debug --soc_version=Ascend310 

atc --framework=5 --model=StarGAN.onnx --output=StarGAN_bs8 --input_format=NCHW \
    --input_shape="real_img:8,3,128,128;attr:8,5" --log=debug --soc_version=Ascend310 

atc --framework=5 --model=StarGAN.onnx --output=StarGAN_bs16 --input_format=NCHW \
    --input_shape="real_img:16,3,128,128;attr:16,5" --log=debug --soc_version=Ascend310 