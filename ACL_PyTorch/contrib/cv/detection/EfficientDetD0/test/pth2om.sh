python3 pth2onnx.py --batch_size=1 --checkpoint=./model/d0.pth --out=./model/d0.onnx
python3 -m onnxsim --input-shape="1,3,512,512" --dynamic-input-shape ./model/d0.onnx ./model/d0_sim.onnx
python3 modify_onnx.py --model=./model/d0_sim.onnx --out=./model/d0_modify.onnx

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest

atc --framework=5 --model=./model/d0_modify.onnx --output=./model/d0 --input_format=NCHW --input_shape="x.1:1,3,512,512" --log=debug --soc_version=Ascend310 --out-nodes="Conv_2043:0;Conv_1973:0;Conv_2057:0;Conv_1987:0;Conv_2071:0;Conv_2001:0;Conv_2085:0;Conv_2015:0;Conv_2099:0;Conv_2029:0"