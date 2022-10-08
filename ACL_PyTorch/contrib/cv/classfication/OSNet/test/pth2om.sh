#!/bin/bash
rm -rf osnet_x1_0.onnx
python3.7 pth2onnx.py osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth osnet_x1_0.onnx 
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf osnet_x1_0_bs1_sim.onnx osnet_x1_0_bs16_sim.onnx osnet_x1_0_bs1.om osnet_x1_0_bs16.om
python3.7 -m onnxsim osnet_x1_0.onnx osnet_x1_0_bs1_sim.onnx --input-shape 1,3,256,128
python3.7 -m onnxsim osnet_x1_0.onnx osnet_x1_0_bs16_sim.onnx --input-shape 16,3,256,128
atc --framework=5 --model=./osnet_x1_0_bs1_sim.onnx --input_format=NCHW --input_shape="image:1,3,256,128" --output=osnet_x1_0_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./osnet_x1_0_bs16_sim.onnx --input_format=NCHW --input_shape="image:16,3,256,128" --output=osnet_x1_0_bs16 --log=debug --soc_version=Ascend310
if [ -f "osnet_x1_0_bs1.om" ] && [ -f "osnet_x1_0_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi
