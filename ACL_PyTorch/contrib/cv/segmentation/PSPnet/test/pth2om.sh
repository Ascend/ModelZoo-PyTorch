#!/bin/bash

rm -rf pspnet_r50-d8_512x512_20k_voc12aug.onnx
rm -rf pspnet_r50-d8_512x512_20k_voc12aug_sim.onnx
python3.7 pytorch2onnx.py ../configs/pspnet/pspnet_r50-d8_512x512_20k_voc12aug.py --checkpoint pspnet_r50-d8_512x512_20k_voc12aug_20200617_010715-52dc5306.pth --output-file pspnet_r50-d8_512x512_20k_voc12aug.onnx --shape 500 500
python3.7 -m onnxsim --input-shape="1,3,500,500" pspnet_r50-d8_512x512_20k_voc12aug.onnx pspnet_r50-d8_512x512_20k_voc12aug_sim.onnx
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf pspnet_r50-d8_512x512_20k_voc12aug_sim_bs1.om pspnet_r50-d8_512x512_20k_voc12aug_sim_bs16.om
atc --framework=5 --model=pspnet_r50-d8_512x512_20k_voc12aug_sim.onnx  --output=pspnet_r50-d8_512x512_20k_voc12aug_sim_bs1 --input_format=NCHW --input_shape=" input:1,3,500,500" --log=debug --soc_version=Ascend310 --input_fp16_nodes=input
atc --framework=5 --model=pspnet_r50-d8_512x512_20k_voc12aug_sim.onnx  --output=pspnet_r50-d8_512x512_20k_voc12aug_sim_bs16 --input_format=NCHW --input_shape=" input:16,3,500,500" --log=debug --soc_version=Ascend310 --input_fp16_nodes=input
if [ -f "pspnet_r50-d8_512x512_20k_voc12aug_bs1.om" ] && [ -f "pspnet_r50-d8_512x512_20k_voc12aug_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi