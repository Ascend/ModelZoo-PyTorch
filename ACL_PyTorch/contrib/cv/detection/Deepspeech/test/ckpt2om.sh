#!/bin/bash

source test/env.sh

currentDir=$(cd "$(dirname "$0")";pwd)/..
rm -rf deepspeech.onnx
python3 $currentDir/ckpt2onnx.py --ckpt_path $currentDir/an4_pretrained_v3.ckpt --out_file deepspeech.onnx
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf deepspeech_bs1.om
atc --framework=5 --model=deepspeech.onnx --input_format=NCHW --input_shape="spect:1,1,161,621;transcript:1" --output=deepspeech_bs1 --log=debug --soc_version=Ascend310

if [ -f "deepspeech_bs1.om" ] ; then
    echo "success"
else
    echo "fail!"
fi
