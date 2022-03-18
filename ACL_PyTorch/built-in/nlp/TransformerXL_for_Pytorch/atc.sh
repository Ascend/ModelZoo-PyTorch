export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest

out_names=$(python3.7 get_out_node.py $1)

atc --model=$1 --framework=5 --output=$2 --input_format=NCHW --log=error --soc_version=Ascend310 --input_fp16_nodes="2;3;4;5;6;7;8;9;10;11;12;mems_i;14" --output_type=FP16 --out_nodes=$out_names

