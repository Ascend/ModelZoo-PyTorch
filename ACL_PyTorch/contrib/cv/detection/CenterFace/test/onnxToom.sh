#!/bin/bash
cd ../
source npu_set_env.sh

atc --framework=5 --model=CenterFace.onnx --input_format=NCHW --input_shape="image:1,3,800,800" --output=CenterFace_bs1 --log=debug --soc_version=Ascend310
