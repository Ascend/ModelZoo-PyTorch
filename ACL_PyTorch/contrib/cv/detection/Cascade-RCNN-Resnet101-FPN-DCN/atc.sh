source /usr/local/Ascend/ascend-toolkit/set_env.sh

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=$1 --framework=5 --output=$2 --input_format=NCHW --input_shape="input:1,3,1216,1216" --log=info --soc_version=Ascend${chip_name} --out_nodes="Concat_1427:0;Reshape_1429:0"