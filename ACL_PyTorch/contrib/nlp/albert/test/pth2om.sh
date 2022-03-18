# encoding=utf-8

mkdir outputs
source env.sh

echo batch_size_1 pth2onnx
python3.7 ./Albert_pth2onnx.py --batch_size=1 --pth_dir=./albert_pytorch/outputs/SST-2/ --onnx_dir=./outputs/

echo batch_size_1 onnxsim
python3.7 -m onnxsim ./outputs/albert_bs1.onnx ./outputs/albert_bs1s.onnx

echo batch_size_1 onnx2om
atc --input_format=ND --framework=5 --model=./outputs/albert_bs1s.onnx --output=./outputs/albert_bs1s --log=error --soc_version=Ascend310 --input_shape="input_ids:1,128;attention_mask:1,128;token_type_ids:1,128" >atc.log


echo batch_size_4 pth2onnx
python3.7 ./Albert_pth2onnx.py --batch_size=4 --pth_dir=./albert_pytorch/outputs/SST-2/ --onnx_dir=./outputs/

echo batch_size_4 onnxsim
python3.7 -m onnxsim ./outputs/albert_bs4.onnx ./outputs/albert_bs4s.onnx

echo batch_size_4 onnx2om
atc --input_format=ND --framework=5 --model=./outputs/albert_bs4s.onnx --output=./outputs/albert_bs4s --log=error --soc_version=Ascend310 --input_shape="input_ids:4,128;attention_mask:4,128;token_type_ids:4,128" >atc.log


echo batch_size_8 pth2onnx
python3.7 ./Albert_pth2onnx.py --batch_size=8 --pth_dir=./albert_pytorch/outputs/SST-2/ --onnx_dir=./outputs/

echo batch_size_8 onnxsim
python3.7 -m onnxsim ./outputs/albert_bs8.onnx ./outputs/albert_bs8s.onnx

echo batch_size_8 onnx2om
atc --input_format=ND --framework=5 --model=./outputs/albert_bs8s.onnx --output=./outputs/albert_bs8s --log=error --soc_version=Ascend310 --input_shape="input_ids:8,128;attention_mask:8,128;token_type_ids:8,128" >atc.log


echo batch_size_16 pth2onnx
python3.7 ./Albert_pth2onnx.py --batch_size=16 --pth_dir=./albert_pytorch/outputs/SST-2/ --onnx_dir=./outputs/

echo batch_size_16 onnxsim
python3.7 -m onnxsim ./outputs/albert_bs16.onnx ./outputs/albert_bs16s.onnx

echo batch_size_16 onnx2om
atc --input_format=ND --framework=5 --model=./outputs/albert_bs16s.onnx --output=./outputs/albert_bs16s --log=error --soc_version=Ascend310 --input_shape="input_ids:16,128;attention_mask:16,128;token_type_ids:16,128" >atc.log


echo batch_size_32 pth2onnx
python3.7 ./Albert_pth2onnx.py --batch_size=32 --pth_dir=./albert_pytorch/outputs/SST-2/ --onnx_dir=./outputs/

echo batch_size_32 onnxsim
python3.7 -m onnxsim ./outputs/albert_bs32.onnx ./outputs/albert_bs32s.onnx

echo batch_size_32 onnx2om
atc --input_format=ND --framework=5 --model=./outputs/albert_bs32s.onnx --output=./outputs/albert_bs32s --log=error --soc_version=Ascend310 --input_shape="input_ids:32,128;attention_mask:32,128;token_type_ids:32,128" >atc.log
