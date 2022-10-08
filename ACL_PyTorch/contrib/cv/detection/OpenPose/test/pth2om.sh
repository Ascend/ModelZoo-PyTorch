#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh

python3.7 OpenPose_pth2onnx.py --checkpoint_path='./weights/checkpoint_iter_370000.pth' --output_name="./output/human-pose-estimation.onnx"

atc --framework=5 --model=./output/human-pose-estimation.onnx --output=./output/human-pose-estimation_bs1 --input_format=NCHW --input_shape="data:1, 3, 368, 640" --log=debug --soc_version=Ascend310

atc --framework=5 --model=./output/human-pose-estimation.onnx --output=./output/human-pose-estimation_bs16 --input_format=NCHW --input_shape="data:16, 3, 368, 640" --log=debug --soc_version=Ascend310

if [ -f "./output/human-pose-estimation_bs1.om" ] && [ -f "./output/human-pose-estimation_bs16.om" ]
then
  echo "success"
else
  echo "fail"
fi

