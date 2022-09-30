#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir onnx_models
python swin_pth2onnx.py --resume=resume/swin_tiny_patch4_window7_224.pth --cfg=Swin-Transformer/configs/swin_tiny_patch4_window7_224.yaml --batch-size=1
python swin_pth2onnx.py --resume=resume/swin_tiny_patch4_window7_224.pth --cfg=Swin-Transformer/configs/swin_tiny_patch4_window7_224.yaml --batch-size=16
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python fix_softmax.py onnx_models/swin_b1.onnx onnx_models/swin_b1_fix.onnx
python fix_softmax.py onnx_models/swin_b16.onnx onnx_models/swin_b16_fix.onnx
atc --framework=5 --model=onnx_models/swin_b1_fix.onnx  --output=swin_b1 --input_format=NCHW --input_shape="image:1,3,224,224" --log=debug --soc_version=Ascend310
if [ -f "swin_b1.om" ] && [ -f "swin_b1.om" ]; then
    echo "success"
else
    echo "fail!"
fi
atc --framework=5 --model=onnx_models/swin_b16_fix.onnx  --output=swin_b16 --input_format=NCHW --input_shape="image:16,3,224,224" --log=debug --soc_version=Ascend310
if [ -f "swin_b16.om" ] && [ -f "swin_b16.om" ]; then
    echo "success"
else
    echo "fail!"
fi