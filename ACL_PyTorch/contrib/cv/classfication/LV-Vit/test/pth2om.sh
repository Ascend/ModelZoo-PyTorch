#! /bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# start
python LV_Vit_pth2onnx.py ./model/model_best.pth.tar ./model/model_best_bs1.onnx 1;
python LV_Vit_pth2onnx.py ./model/model_best.pth.tar ./model/model_best_bs16.onnx 16;

python -m onnxsim --input-shape="1,3,224,224" ./model/model_best_bs1.onnx ./model/model_best_bs1_sim.onnx;
python -m onnxsim --input-shape="16,3,224,224" ./model/model_best_bs16.onnx ./model/model_best_bs16_sim.onnx;

atc --framework=5 --model=./model/model_best_bs1_sim.onnx --output=./model/model_best_bs1_sim --input_format=NCHW --input_shape="image:1,3,224,224" --log=debug --soc_version=Ascend310;
atc --framework=5 --model=./model/model_best_bs16_sim.onnx --output=./model/model_best_bs16_sim --input_format=NCHW --input_shape="image:16,3,224,224" --log=debug --soc_version=Ascend310;
