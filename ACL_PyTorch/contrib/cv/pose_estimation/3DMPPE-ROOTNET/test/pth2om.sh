#!/bin/bash

rm -rf 3DMPPE-ROOTNET.onnx
python3.7 3DMPPE-ROOTNET_pth2onnx.py snapshot_6.pth.tar 3DMPPE-ROOTNET.onnx
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf 3DMPPE-ROOTNET_bs1.om 3DMPPE-ROOTNET_bs16.om
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --framework=5 --model=3DMPPE-ROOTNET.onnx --output=3DMPPE-ROOTNET_bs1 --input_format=NCHW --input_shape="image:1,3,256,256;cam_param:1,1" --log=error --soc_version=Ascend310
atc --framework=5 --model=3DMPPE-ROOTNET.onnx --output=3DMPPE-ROOTNET_bs16 --input_format=NCHW --input_shape="image:16,3,256,256;cam_param:16,1" --log=error --soc_version=Ascend310

if [ -f "3DMPPE-ROOTNET_bs1.om" ] && [ -f "3DMPPE-ROOTNET_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi