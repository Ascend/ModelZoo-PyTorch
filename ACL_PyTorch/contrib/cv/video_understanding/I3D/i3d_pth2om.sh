python3.8 tools/pytorch2onnx.py configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py checkpoints/i3d_r50_32x2x1_100e_kinetics400_rgb_20200614-c25ef9a4.pth --shape 1 30 3 32 256 256 --verify --show --output i3d.onnx --opset-version 11

#!/bin/bash

export install_path="/usr/local/Ascend/ascend-toolkit/latest"
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_SLOG_PRINT_TO_STDOUT=1
${install_path}/atc/bin/atc --framework=5 --output=./i3d_bs1  --input_format=NCHW  --soc_version=Ascend310 \
    --model=./i3d.onnx --input_shape="0:1,30,3,32,256,256"

