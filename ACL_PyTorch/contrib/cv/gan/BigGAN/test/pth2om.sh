#!/bin/bash

source env.sh

rm -rf biggan.onnx
python3.7 biggan_pth2onnx.py --source 'G_ema.pth' --target './biggan.onnx'
if [ -f "biggan.onnx" ]; then
  echo "==> 1. creating onnx model successfully."
else
  echo "onnx export failed"
  exit -1
fi

python3.7 clip_edit.py --input-model './biggan.onnx' --output-model './biggan.onnx'
if [ $? != 0 ]; then
    echo "Clip max initializer fail!"
    exit -1
fi
echo '==> 2. adding Clip max initializer'

rm -rf biggan_sim_bs1.onnx biggan_sim_bs16.onnx
python3.7 -m onnxsim './biggan.onnx' './biggan_sim_bs1.onnx' --input-shape "noise:1,1,20" "label:1,5,148"
python3.7 -m onnxsim './biggan.onnx' './biggan_sim_bs16.onnx' --input-shape "noise:16,1,20" "label:16,5,148"
if [ -f "biggan_sim_bs1.onnx" ] && [ -f "biggan_sim_bs16.onnx" ]; then
  echo "==> 3. creating onnx sim model successfully."
else
  echo "sim_onnx export failed"
  exit -1
fi

rm -rf biggan_sim_bs1.om biggan_sim_bs16.om
atc --framework=5 --model=./biggan_sim_bs1.onnx --output=./biggan_sim_bs1 --input_format=ND --input_shape="noise:1,1,20;label:1,5,148" --log=error --soc_version=Ascend310
atc --framework=5 --model=./biggan_sim_bs16.onnx --output=./biggan_sim_bs16 --input_format=ND --input_shape="noise:16,1,20;label:16,5,148" --log=error --soc_version=Ascend310
if [ -f "biggan_sim_bs1.om" ] && [ -f "biggan_sim_bs16.om" ]; then
  echo "==> 4. creating om model successfully."
else
  echo "sim_om export failed"
fi
echo "==> 5. Done."
