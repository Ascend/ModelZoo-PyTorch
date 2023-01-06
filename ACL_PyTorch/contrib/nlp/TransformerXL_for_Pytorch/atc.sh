names=$(python3 get_node_name.py $1)
in_names=${names%end*}
out_names=${names#*end}

atc --model=$1 --framework=5 --output=$2 --input_format=ND --log=error --soc_version=$3  \
--input_fp16_nodes=${in_names} \
--output_type=FP16 --out_nodes=${out_names} \
