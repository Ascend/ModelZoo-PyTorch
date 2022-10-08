# CANN安装路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc --framework=5 --model=./FOTS_bs8.onnx --output=./FOTS_bs8 --input_format=NCHW --input_shape="image:8,3,1248,2240" --log=debug --soc_version=Ascend310