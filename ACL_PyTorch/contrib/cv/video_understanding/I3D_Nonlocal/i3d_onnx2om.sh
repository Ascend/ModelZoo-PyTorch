#!/bin/bash
atc --framework=5 --output=i3d_nl_dot  --input_format=NCHW  --soc_version=Ascend310P3 --model=i3d_nl_dot.onnx --input_shape="0:1,10,3,32,256,256"


