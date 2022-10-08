python3.7 TinyBERT_postprocess_data.py \
--model ./SST-2_model \
--do_lower_case \
--max_seq_length 64 \
--data_dir ./glue_dir/SST-2 \
--result_dir ./result/$1 \
--inference_tool "ais_infer"