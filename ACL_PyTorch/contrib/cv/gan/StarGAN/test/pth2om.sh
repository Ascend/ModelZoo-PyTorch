#!/bin/bash
PTH_FILE=$1

source /usr/local/Ascend/ascend-toolkit/set_env.sh

python3.7 pth2onnx.py --input_file $PTH_FILE --output_file './StarGAN.onnx'
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

atc --framework=5 --model=StarGAN.onnx --output=StarGAN_bs1 --input_format=NCHW \
    --input_shape="real_img:1,3,128,128;attr:1,5" --log=debug --soc_version=Ascend310 

atc --framework=5 --model=StarGAN.onnx --output=StarGAN_bs4 --input_format=NCHW \
    --input_shape="real_img:4,3,128,128;attr:4,5" --log=debug --soc_version=Ascend310 

atc --framework=5 --model=StarGAN.onnx --output=StarGAN_bs8 --input_format=NCHW \
    --input_shape="real_img:8,3,128,128;attr:8,5" --log=debug --soc_version=Ascend310 

atc --framework=5 --model=StarGAN.onnx --output=StarGAN_bs16 --input_format=NCHW \
    --input_shape="real_img:16,3,128,128;attr:16,5" --log=debug --soc_version=Ascend310 

atc --framework=5 --model=StarGAN.onnx --output=StarGAN_bs32 --input_format=NCHW \
    --input_shape="real_img:32,3,128,128;attr:32,5" --log=debug --soc_version=Ascend310 

atc --framework=5 --model=StarGAN.onnx --output=StarGAN_bs64 --input_format=NCHW \
    --input_shape="real_img:64,3,128,128;attr:64,5" --log=debug --soc_version=Ascend310 
