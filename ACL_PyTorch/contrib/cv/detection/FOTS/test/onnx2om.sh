# CANN安装路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc --framework=5 --model=./FOTS_bs1.onnx --output=./FOTS_bs1_auto --input_format=NCHW --input_shape="image:1,3,1248,2240" --log=debug --soc_version=Ascend310 