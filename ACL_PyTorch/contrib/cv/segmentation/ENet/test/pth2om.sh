#!/bin/bash

rm -rf models/enet_citys_910_bs1.onnx models/enet_citys_910_bs16.onnx
python3.7 ENet_pth2onnx.py --input-file models/enet_citys.pth --output-file models/enet_citys_910_bs1.onnx --batch-size 1
python3.7 ENet_pth2onnx.py --input-file models/enet_citys.pth --output-file models/enet_citys_910_bs16.onnx --batch-size 16

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python3.7 -m onnxsim --skip-fuse-bn --input-shape="1,3,480,480" --dynamic-input-shape models/enet_citys_910_bs1.onnx models/enet_citys_910_bs1_sim.onnx
python3.7 -m onnxsim --skip-fuse-bn --input-shape="16,3,480,480" --dynamic-input-shape models/enet_citys_910_bs16.onnx models/enet_citys_910_bs16_sim.onnx

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

source env.sh
rm -rf models/enet_citys_910_bs1.om models/enet_citys_910_bs16.om
atc --framework=5 --model=models/enet_citys_910_bs1_sim.onnx --output=models/enet_citys_910_bs1 --input_format=NCHW --input_shape="image:1,3,480,480" --log=info --soc_version=Ascend310
atc --framework=5 --model=models/enet_citys_910_bs16_sim.onnx --output=models/enet_citys_910_bs16 --input_format=NCHW --input_shape="image:16,3,480,480" --log=info --soc_version=Ascend310
if [ -f "models/enet_citys_910_bs1.om" ] && [ -f "models/enet_citys_910_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi

# python3.7 -m onnxsim --input-shape="1,3,480,480" --dynamic-input-shape test/models/enet_citys_910_bs1.onnx test/models/enet_citys_910_bs1_sim.onnx