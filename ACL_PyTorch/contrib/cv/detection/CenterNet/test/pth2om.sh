#!/bin/bash
python3.7 CenterNet_pth2onnx.py ctdet_coco_dla_2x.pth CenterNet.onnx && source /usr/local/Ascend/ascend-toolkit/set_env.sh
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

atc --framework=5 --model=CenterNet.onnx  --output=CenterNet_bs1_310P --input_format=NCHW --input_shape="actual_input:1,3,512,512" --out_nodes="Conv_1120:0;Conv_1123:0;Conv_1126:0" --log=info --soc_version=Ascend710
atc --framework=5 --model=CenterNet.onnx  --output=CenterNet_bs32_310P --input_format=NCHW --input_shape="actual_input:32,3,512,512" --out_nodes="Conv_1120:0;Conv_1123:0;Conv_1126:0" --log=info --soc_version=Ascend710
if [ -f "CenterNet_bs1_310P.om" ] && [ -f "CenterNet_bs32_310P.om" ]; then
    echo "success"
else
    echo "fail!"
fi