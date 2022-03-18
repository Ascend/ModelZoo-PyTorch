python3.7 TinyBERT_preprocess_data.py \
--model ./SST-2_model \
--data_dir ./glue_dir/SST-2 \
--max_seq_length 64 \
--eval_batch_size 1 \
--do_lower_case \
--inference_tool "benchmark"