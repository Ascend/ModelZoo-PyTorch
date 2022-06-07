#!/bin/bash

# variables for transition from .pdparams to .onnx
config_path=PaddleSeg/configs/ocrnet/ocrnet_hrnetw18_cityscapes_1024x512_160k.yml
pd_model=pd_model
pdparams_path=model.pdparams
onnx_path=ocrnet.onnx


rm -rf ./${pd_model}
python PaddleSeg/export.py --save_dir ${pd_model} --model_path ${pdparams_path} --config ${config_path}
paddle2onnx --model_dir ${pd_model} --model_filename ${pd_model}/model.pdmodel --params_filename ${pd_model}/model.pdiparams --save_file ${onnx_path} --opset_version 11

rm -rf ./om
mkdir om

rm -rf ./onnx
mkdir onnx

for batch_size in 1 4 8 16
do
	python -m onnxsim ocrnet.onnx onnx/ocrnet_bs${batch_size}.onnx --input-shape="x:${batch_size},3,1024,2048" --skip-fuse-bn
	python optimize_onnx.py onnx/ocrnet_bs${batch_size}.onnx onnx/ocrnet_optimize_bs${batch_size}.onnx
	
done

source /usr/local/Ascend/ascend-toolkit/set_env.sh


atc --framework=5 --model=onnx/ocrnet_optimize_bs1.onnx --output=om/ocrnet_optimize_bs1 --input_format=NCHW --input_shape="x:1,3,1024,2048" --soc_version=Ascend710 --log=debug
atc --framework=5 --model=onnx/ocrnet_optimize_bs4.onnx --output=om/ocrnet_optimize_bs4 --input_format=NCHW --input_shape="x:4,3,1024,2048" --soc_version=Ascend710 --log=debug
atc --framework=5 --model=onnx/ocrnet_optimize_bs8.onnx --output=om/ocrnet_optimize_bs8 --input_format=NCHW --input_shape="x:8,3,1024,2048" --soc_version=Ascend710 --log=debug
atc --framework=5 --model=onnx/ocrnet_optimize_bs16.onnx --output=om/ocrnet_optimize_bs16 --input_format=NCHW --input_shape="x:16,3,1024,2048" --soc_version=Ascend710 --log=debug
