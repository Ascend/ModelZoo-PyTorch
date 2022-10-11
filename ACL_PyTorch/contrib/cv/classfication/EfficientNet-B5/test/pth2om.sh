#!/bin/bash
rm -rf efficientnetb5.onnx
python3.7 efficientnetb5_pth2onnx.py efficientnetb5.pyth efficientnetb5_dds_8gpu.yaml  efficientnetb5.onnx
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf efficientnetb5_bs1.om efficientnetb5_bs16.om
atc --framework=5 --model=efficientnetb5.onnx --output=efficientnetb5_bs1 --input_format=NCHW --input_shape="image:1,3,456,456" --auto_tune_mode="RL,GA" --log=debug --soc_version=Ascend310
atc --framework=5 --model=efficientnetb5.onnx --output=efficientnetb5_bs16 --input_format=NCHW --input_shape="image:16,3,456,456" --auto_tune_mode="RL,GA" --log=debug --soc_version=Ascend310
if [ -f "efficientnetb5_bs1.om" ] && [ -f "efficientnetb5_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi
