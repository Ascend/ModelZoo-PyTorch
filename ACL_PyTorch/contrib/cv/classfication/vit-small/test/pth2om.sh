#!/bin/bash
rm -rf vit_small_patch16_224.onnx 
python3.7 vit_small_patch16_224_pth2onnx.py ./S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz ./vit_small_patch16_224.onnx 
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf vit_small_patch16_224_sim_1.onnx vit_small_patch16_224_sim_16.onnx
python3.7 -m onnxsim --input-shape="1,3,224,224" vit_small_patch16_224.onnx vit_small_patch16_224_sim_1.onnx
python3.7 -m onnxsim --input-shape="16,3,224,224" vit_small_patch16_224.onnx vit_small_patch16_224_sim_16.onnx
if [ -f "vit_small_patch16_224_sim_1.onnx" ] && [ -f "vit_small_patch16_224_sim_16.onnx" ]; then
    echo "onnxsim success"
else
    echo "onnxsim fail!"
fi

rm -rf vit_small_patch16_224_bs1_sim.om vit_small_patch16_224_bs16_sim.om
source env.sh
atc --framework=5 --model=vit_small_patch16_224_sim_1.onnx --output=vit_small_patch16_224_bs1_sim --input_format=NCHW --input_shape="image:1,3,224,224" --log=error --soc_version=Ascend310 --enable_small_channel=1 --auto_tune_mode="RL,GA"
atc --framework=5 --model=vit_small_patch16_224_sim_16.onnx --output=vit_small_patch16_224_bs16_sim --input_format=NCHW --input_shape="image:16,3,224,224" --log=error --soc_version=Ascend310 --enable_small_channel=1 --auto_tune_mode="RL,GA"

if [ -f "vit_small_patch16_224_bs1_sim.om" ] && [ -f "vit_small_patch16_224_bs16_sim.om" ]; then
    echo "onnx2om success"
else
    echo "onnx2om fail!"
fi

