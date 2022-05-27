#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe:$PYTHONPATH
export PATH=/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin:$PATH
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
export TOOLCHAIN_HOME=/usr/local/Ascend/ascend-toolkit/latest/toolkit
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_SLOG_PRINT_TO_STDOUT=1

rm -rf CenterNet_bs1_710.om CenterNet_bs32_710.om
atc --framework=5 --model=CenterNet.onnx  --output=CenterNet_bs1_710 --input_format=NCHW --input_shape="actual_input:1,3,512,512" --out_nodes="Conv_1120:0;Conv_1123:0;Conv_1126:0" --log=info --soc_version=Ascend710
atc --framework=5 --model=CenterNet.onnx  --output=CenterNet_bs1_710 --input_format=NCHW --input_shape="actual_input:32,3,512,512" --out_nodes="Conv_1120:0;Conv_1123:0;Conv_1126:0" --log=info --soc_version=Ascend710
if [ -f "CenterNet_bs1_710.om" ] && [ -f "CenterNet_bs32_710.om" ]; then
    echo "success"
else
    echo "fail!"
fi

