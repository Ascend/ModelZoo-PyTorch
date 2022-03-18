#!/bin/bash
install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_SLOG_PRINT_TO_STDOUT=1

rm -rf CenterNet_bs1.om CenterNet_bs16.om
atc --framework=5 --model=CenterNet.onnx  --output=CenterNet_bs1 --input_format=NCHW --input_shape="actual_input:1,3,512,512" --out_nodes="Conv_949:0;Conv_952:0;Conv_955:0" --log=info --soc_version=Ascend310
atc --framework=5 --model=CenterNet.onnx  --output=CenterNet_bs16 --input_format=NCHW --input_shape="actual_input:16,3,512,512" --out_nodes="Conv_949:0;Conv_952:0;Conv_955:0" --log=info --soc_version=Ascend310
if [ -f "CenterNet_bs1.om" ] && [ -f "CenterNet_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi
