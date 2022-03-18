#!/bin/bash
rm -rf realesrgan-x4.onnx
python ./Real-ESRGAN/scripts/pytorch2onnx.py
mv realesrgan-x4.onnx realesrgan-x4-b1-64.onnx
patch -p0 ./Real-ESRGAN/scripts/pytorch2onnx.py < pth2onnx.patch
python ./Real-ESRGAN/scripts/pytorch2onnx.py
rm -rf realesrgan-x4_bs1.om realesrgan-x4_bs4.om realesrgan-x4_bs8.om realesrgan-x4_bs16.om realesrgan-x4_bs32.om realesrgan_bs1-220.om
source ./env.sh
atc --framework=5 --model=realesrgan-x4-b1-220.onnx --output=realesrgan_bs1-220 --input_format=NCHW --input_shape="input.1:1,3,220,220" --log=debug --soc_version=Ascend310
atc --framework=5 --model=realesrgan-x4-b1-64.onnx --output=realesrgan_bs1 --input_format=NCHW --input_shape="input.1:1,3,64,64" --log=debug --soc_version=Ascend310
atc --framework=5 --model=realesrgan-x4-b1-64.onnx --output=realesrgan_bs4 --input_format=NCHW --input_shape="input.1:4,3,64,64" --log=debug --soc_version=Ascend310
atc --framework=5 --model=realesrgan-x4-b1-64.onnx --output=realesrgan_bs8 --input_format=NCHW --input_shape="input.1:8,3,64,64" --log=debug --soc_version=Ascend310
atc --framework=5 --model=realesrgan-x4-b1-64.onnx --output=realesrgan_bs16 --input_format=NCHW --input_shape="input.1:16,3,64,64" --log=debug --soc_version=Ascend310
atc --framework=5 --model=realesrgan-x4-b1-64.onnx --output=realesrgan_bs32 --input_format=NCHW --input_shape="input.1:32,3,64,64" --log=debug --soc_version=Ascend310
if [ -f "realesrgan_bs1.om" ] &&[ -f "realesrgan_bs1-220.om" ]&& [ -f "realesrgan_bs4.om" ]&&[ -f "realesrgan_bs8.om" ] && [ -f "realesrgan_bs16.om" ]&&[ -f "realesrgan_bs32.om" ]; then
    echo "success"
else
    echo "fail!"
fi
