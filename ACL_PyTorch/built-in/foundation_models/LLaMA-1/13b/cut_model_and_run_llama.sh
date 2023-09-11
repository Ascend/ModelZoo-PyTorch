#!/bin/bash
export HCCL_OP_BASE_FFTS_MODE_ENABLE=1
export ACLTRANSFORMER_PLAN_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=110

# custom model path
input_dir="/data/models/llama-13b"
output_dir="/data/models/llama-13b-part_model_2_test"

world_size_=2
cut_row_keys_=['q_proj','k_proj','v_proj','gate_proj','up_proj']
cut_col_keys_=['o_proj','down_proj']

# if model has already been cutted, then run the model; if not, cut the model first
if test -d "$output_dir";
then
    echo "Weight directory exists, runing......"
    torchrun --nproc_per_node 2 run_llama1_13b_parallel.py --load_path $output_dir
else 
    echo "Cutted Weight directory does not exist, cuting the weight......"
    python ./cut_model_util.py --input_path $input_dir --output_path $output_dir --world_size $world_size_ --cut_row_keys $cut_row_keys_ --cut_col_keys $cut_col_keys_
fi