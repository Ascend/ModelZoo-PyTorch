#!/bin/bash

# 生成onnx文件
rm -rf C3D.onnx
cd mmaction2-master
python tools/pytorch2onnx.py configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py checkpoints/C3D.pth --shape 1 10 3 16 112 112 --verify --softmax
mv tmp.onnx C3D.onnx
mv ./C3d.onnx /root/c3d/ 
# 配置环境变量
cd /root/c3d
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 生成om文件
rm -rf C3D.om
atc --framework=5 --model=C3D.onnx --output=C3D --input_format=ND --input_shape="image:1,10,3,16,112,112" --log=debug --soc_version=Ascend310
if [ -f "C3D.om" ]; then
    echo "success"
else
    echo "fail!"
fi
