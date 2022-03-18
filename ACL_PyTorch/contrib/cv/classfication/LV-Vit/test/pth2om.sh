#! /bin/bash
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest

# start
python LV_Vit_pth2onnx.py ./model/model_best.pth.tar ./model/model_best_bs1.onnx 1;
python LV_Vit_pth2onnx.py ./model/model_best.pth.tar ./model/model_best_bs16.onnx 16;

python -m onnxsim --input-shape="1,3,224,224" ./model/model_best_bs1.onnx ./model/model_best_bs1_sim.onnx;
python -m onnxsim --input-shape="16,3,224,224" ./model/model_best_bs16.onnx ./model/model_best_bs16_sim.onnx;

atc --framework=5 --model=./model/model_best_bs1_sim.onnx --output=./model/model_best_bs1_sim --input_format=NCHW --input_shape="image:1,3,224,224" --log=debug --soc_version=Ascend310;
atc --framework=5 --model=./model/model_best_bs16_sim.onnx --output=./model/model_best_bs16_sim --input_format=NCHW --input_shape="image:16,3,224,224" --log=debug --soc_version=Ascend310;
