#!/bin/bash
source test/env.sh
git clone --branch v0.4.5 https://github.com/rwightman/pytorch-image-models.git
cd pytorch-image-models  
patch -p1 < ../TResNet.patch
cd ..  
python3 TResNet_pth2onnx.py model_best.pth.tar tresnet_m.onnx
atc --framework=5 --model=tresnet_m.onnx --output=tresnet_patch16_224_bs1 --input_format=NCHW --input_shape="image:1,3,224,224" --log=debug --soc_version=Ascend310
atc --framework=5 --model=tresnet_m.onnx --output=tresnet_patch16_224_bs16 --input_format=NCHW --input_shape="image:16,3,224,224" --log=debug --soc_version=Ascend310
