python3 pth2onnx.py --batch_size=1 --checkpoint=./model/d7.pth --out=./model/d7.onnx 
python3 -m onnxsim --input-shape="1,3,1536,1536" --dynamic-input-shape ./model/d7.onnx ./model/d7_sim.onnx
python3 modify_onnx.py --model=./model/d7_sim.onnx --node=3080 --out=./model/d7_modify.onnx

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest

atc --framework=5 --model=./model/d7_modify.onnx --output=./model/d7 --input_format=NCHW --input_shape="x.1:1,3,1536,1536" --log=debug --soc_version=Ascend310