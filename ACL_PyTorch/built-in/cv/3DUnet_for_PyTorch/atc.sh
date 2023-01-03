source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --model=$1 --framework=5 --output=$2 --input_format=ND --input_shape="input:1,4,224,224,160" --log=info --soc_version=$3 --out_nodes="Conv_80:0"