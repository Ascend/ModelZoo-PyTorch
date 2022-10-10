#!/bin/bash

echo 'pth -> onnx dybs'
rm -rf wideresnet_dybs.onnx
python3.7 Wseg_pth2onnx.py "wseg/configs/voc_resnet38.yaml" './snapshots/model_enc_e020Xs0.928.pth'
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "start fix onnx!"
python3.7 fix_softmax_transpose.py ./wideresnet_dybs.onnx ./wideresnet_dybs_fix.onnx

rm -rf wideresnet_bs1.om wideresnet_bs4.om
source /usr/local/Ascend/ascend-toolkit/set_env.sh
echo 'onnx -> om batch1'
atc --model=./wideresnet_dybs_fix.onnx --framework=5 --output=wideresnet_bs1 --input_format=NCHW --input_shape="image:1,3,1024,1024" --log=debug --soc_version=Ascend${chip_name}
echo 'onnx -> om batch4'
atc --model=./wideresnet_dybs_fix.onnx --framework=5 --output=wideresnet_bs4 --input_format=NCHW --input_shape="image:4,3,1024,1024" --log=debug --soc_version=Ascend${chip_name}
if [ -f "wideresnet_bs1.om" ] && [ -f "wideresnet_bs4.om" ]; then
    echo "success"
else
    echo "fail!"
fi
rm -rf logs
rm -rf kernel_meta