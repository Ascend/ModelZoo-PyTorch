#!/bin/bash
rm -rf hrnet_w18.onnx
source env.sh
python3.7 hrnet_pth2onnx.py --cfg ./HRNet-Image-Classification/experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml --input model_best.pth.tar --output hrnet_w18.onnx
#source env.sh
rm -rf hrnet_w18_bs1.om hrnet_w18_bs16.om
atc --framework=5 --model=./hrnet_w18.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=hrnet_w18_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./hrnet_w18.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=hrnet_w18_bs16 --log=debug --soc_version=Ascend310
if [ -f "hrnet_w18_bs1.om" ] && [ -f "hrnet_w18_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi
