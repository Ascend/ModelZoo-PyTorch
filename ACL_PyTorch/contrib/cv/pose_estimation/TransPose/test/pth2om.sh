#!/bin/bash
set -eu
rm -rf models/*.onnx
python TransPose_pth2onnx.py --cfg ./TransPose/experiments/transpose_r/TP_R_256x192_d256_h1024_enc3_mh8.yaml --weights models/tp_r_256x192_enc3_d256_h1024_mh8.pth
python TransPose_pth2onnx.py --cfg ./TransPose/experiments/transpose_r/TP_R_256x192_d256_h1024_enc3_mh8.yaml --weights models/tp_r_256x192_enc3_d256_h1024_mh8.pth --bs 16
python -m onnxsim models/tp_r_256x192_enc3_d256_h1024_mh8_bs1.onnx models/tp_r_256x192_enc3_d256_h1024_mh8_bs1_sim.onnx
python -m onnxsim models/tp_r_256x192_enc3_d256_h1024_mh8_bs16.onnx models/tp_r_256x192_enc3_d256_h1024_mh8_bs16_sim.onnx

if [ $? != 0 ]; then
    echo "fail!"
    exit 0
fi

rm -rf models/*.om
source env.sh
atc --framework=5 --model=models/tp_r_256x192_enc3_d256_h1024_mh8_bs1_sim.onnx --output=models/tp_r_256x192_enc3_d256_h1024_mh8_bs1 --input_format=NCHW --input_shape="input:1,3,256,192" --fusion_switch_file=fusion_switch.cfg --soc_version=Ascend310
atc --framework=5 --model=models/tp_r_256x192_enc3_d256_h1024_mh8_bs16_sim.onnx --output=models/tp_r_256x192_enc3_d256_h1024_mh8_bs16 --input_format=NCHW --input_shape="input:16,3,256,192" --fusion_switch_file=fusion_switch.cfg --soc_version=Ascend310

if [ -f "models/tp_r_256x192_enc3_d256_h1024_mh8_bs1.om" ] && [ -f "models/tp_r_256x192_enc3_d256_h1024_mh8_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi