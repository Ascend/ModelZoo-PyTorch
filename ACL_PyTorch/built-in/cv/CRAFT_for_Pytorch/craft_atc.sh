source /usr/local/Ascend/ascend-toolkit/set_env.sh 
atc --model=craft.onnx --framework=5 --output=craft --input_format=NCHW \
--input_shape="input:1,3,640,640" --log=error --soc_version=$1
