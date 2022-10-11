#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

mkdir -p onnx_sim
mkdir -p om

python3.7 pytorch2onnx.py mmaction2/configs/recognition/tsm/tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb.py ./tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb_20200724-d8ad84d2.pth --output-file=tsm_nl.onnx --softmax --verify --show --shape 1 8 3 224 224

python3.7 -m onnxsim --input-shape="1,8,3,224,224" tsm_nl.onnx onnx_sim/tsm_nl_bs1.onnx
python3.7 -m onnxsim --input-shape="16,8,3,224,224" tsm_nl.onnx onnx_sim/tsm_nl_bs16.onnx

atc --model=onnx_sim/tsm_nl_bs1.onnx --framework=5 --output=om/tsm_nl_bs1 --input_format=ND --input_shape="video:1,8,3,224,224" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"
atc --model=onnx_sim/tsm_nl_bs16.onnx --framework=5 --output=om/tsm_nl_bs16 --input_format=ND --input_shape="video:16,8,3,224,224" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"

if [ -f "om/tsm_nl_bs1.om" ] && [ -f "om/tsm_nl_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi
#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

mkdir -p onnx_sim
mkdir -p om

python3.7 pytorch2onnx.py mmaction2/configs/recognition/tsm/tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb.py ./tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb_20200724-d8ad84d2.pth --output-file=tsm_nl.onnx --softmax --verify --show --shape 1 8 3 224 224

python3.7 -m onnxsim --input-shape="1,8,3,224,224" tsm_nl.onnx onnx_sim/tsm_nl_bs1.onnx
python3.7 -m onnxsim --input-shape="16,8,3,224,224" tsm_nl.onnx onnx_sim/tsm_nl_bs16.onnx

atc --model=onnx_sim/tsm_nl_bs1.onnx --framework=5 --output=om/tsm_nl_bs1 --input_format=ND --input_shape="video:1,8,3,224,224" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"
atc --model=onnx_sim/tsm_nl_bs16.onnx --framework=5 --output=om/tsm_nl_bs16 --input_format=ND --input_shape="video:16,8,3,224,224" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"

if [ -f "om/tsm_nl_bs1.om" ] && [ -f "om/tsm_nl_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi
