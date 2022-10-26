python3 pth2onnx.py --batch_size=1 --checkpoint=./model/d7.pth --out=./model/d7.onnx 
python3 -m onnxsim --input-shape="1,3,1536,1536" --dynamic-input-shape ./model/d7.onnx ./model/d7_sim.onnx
python3 modify_onnx.py --model=./model/d7_sim.onnx --node=3080 --out=./model/d7_modify.onnx

source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc --framework=5 --model=./model/d7_modify.onnx --output=./model/d7 --input_format=NCHW --input_shape="x.1:1,3,1536,1536" --log=debug --soc_version=Ascend310