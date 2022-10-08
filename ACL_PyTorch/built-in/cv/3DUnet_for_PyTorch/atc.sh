source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_SLOG_PRINT_TO_STDOUT=1

atc --model=$1 --framework=5 --output=$2 --input_format=ND --input_shape="input:1,4,224,224,160" --log=info --soc_version=Ascend310 --out_nodes="Conv_80:0"