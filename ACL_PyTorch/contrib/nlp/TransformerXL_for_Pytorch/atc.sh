source /usr/local/Ascend/ascend-toolkit/set_env.sh

out_names=$(python3.8 get_out_node.py $1)

atc --model=$1 --framework=5 --output=$2 --input_format=ND --log=error --soc_version=$3  \
--input_fp16_nodes="2;3;4;5;6;7;8;9;10;11;12;mems_i;14" \
--output_type=FP16 --out_nodes=$out_names \
