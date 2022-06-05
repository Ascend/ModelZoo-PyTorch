#!/bin/bash

rm -rf ./pd_model
python PaddleSeg/export.py --save_dir pd_model --model_path model.pdparams --config PaddleSeg/configs/ocrnet/ocrnet_hrnetw18_cityscapes_1024x512_160k.yml
paddle2onnx --model_dir pd_model --model_filename pd_model/model.pdmodel --params_filename pd_model/model.pdiparams --save_file ocrnet.onnx --opset_version 11

rm -rf ./om
mkdir om

rm -rf ./onnx
mkdir onnx

python -m onnxsim ocrnet.onnx onnx/ocrnet_bs1.onnx --input-shape="x:1,3,1024,2048" --skip-fuse-bn
python -m onnxsim ocrnet.onnx onnx/ocrnet_bs4.onnx --input-shape="x:4,3,1024,2048" --skip-fuse-bn
python -m onnxsim ocrnet.onnx onnx/ocrnet_bs8.onnx --input-shape="x:8,3,1024,2048" --skip-fuse-bn
python -m onnxsim ocrnet.onnx onnx/ocrnet_bs16.onnx --input-shape="x:16,3,1024,2048" --skip-fuse-bn

python optimize_onnx.py onnx/ocrnet_bs1.onnx onnx/ocrnet_optimize_bs1.onnx
python optimize_onnx.py onnx/ocrnet_bs4.onnx onnx/ocrnet_optimize_bs4.onnx
python optimize_onnx.py onnx/ocrnet_bs8.onnx onnx/ocrnet_optimize_bs8.onnx
python optimize_onnx.py onnx/ocrnet_bs16.onnx onnx/ocrnet_optimize_bs16.onnx

#onnx to om
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TUNE_BANK_PATH=./ocrnet_aoe_bs1
export TE_PARALLEL_COMPILER=8
export REPEAT_TUNE=False

chmod 777 ocrnet_aoe_bs1

atc --framework=5 --model=onnx/ocrnet_optimize_bs1.onnx --output=om/ocrnet_optimize_bs1 --input_format=NCHW --input_shape="x:1,3,1024,2048" --soc_version=Ascend710 --log=debug
atc --framework=5 --model=onnx/ocrnet_optimize_bs4.onnx --output=om/ocrnet_optimize_bs4 --input_format=NCHW --input_shape="x:4,3,1024,2048" --soc_version=Ascend710 --log=debug
atc --framework=5 --model=onnx/ocrnet_optimize_bs8.onnx --output=om/ocrnet_optimize_bs8 --input_format=NCHW --input_shape="x:8,3,1024,2048" --soc_version=Ascend710 --log=debug
atc --framework=5 --model=onnx/ocrnet_optimize_bs16.onnx --output=om/ocrnet_optimize_bs16 --input_format=NCHW --input_shape="x:16,3,1024,2048" --soc_version=Ascend710 --log=debug
