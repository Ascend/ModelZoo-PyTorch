#! /bin/bash

for para in $*
do
    if [[ $para == --batch_size* ]]; then
        batch_size=`echo ${para#*=}`
    fi
    if [[ $para == --soc_version* ]]; then
        soc_version=`echo ${para#*=}`
    fi
done

if [ -f "fcenet_dynamicbs.onnx" ]; then
    echo "onnx has existed!"
else
    python ./pytorch2onnx.py \
    ./mmocr/configs/textdet/fcenet/fcenet_r50_fpn_1500e_icdar2015.py \
    ./fcenet_r50_fpn_1500e_icdar2015_20211022-daefb6ed.pth \
    det \
    ./mmocr/data/icdar2015/imgs/test/img_1.jpg \
    --dynamic-export \
    --output-file ./fcenet_dynamicbs.onnx
fi

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -f fcenet.om
source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc --framework=5 --model=./fcenet_dynamicbs.onnx --output=./fcenet_bs$batch_size  --input_format=NCHW \
--input_shape="input:$batch_size,3,1280,2272" --log=error --soc_version=${soc_version}

if [ -f "fcenet_bs$batch_size.om" ]; then
    echo "success"
else
    echo "fail!"
fi