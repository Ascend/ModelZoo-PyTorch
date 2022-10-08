#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# bs1
python3.7.5 pthtar2onnx.py 1 model_lincls_best.pth.tar
# bs16
python3.7.5 pthtar2onnx.py 16 model_lincls_best.pth.tar

atc --framework=5 --model=moco-v2-bs1.onnx --output=moco-v2-atc-bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,224,224" --log=info --soc_version=Ascend310
atc --framework=5 --model=moco-v2-bs16.onnx --output=moco-v2-atc-bs16 --input_format=NCHW --input_shape="actual_input_1:16,3,224,224" --log=info --soc_version=Ascend310

if [ -f "moco-v2-atc-bs1.om" ] && [ -f "moco-v2-atc-bs1.om" ]
then
    echo "success"
else
    echo "fail!"
fi
