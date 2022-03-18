python3.7 pth2onnx.py \
--output_file ./TinyBERT.onnx \
--data_dir ./glue_dir/SST-2 \
--max_seq_length 64 \
--eval_batch_size 1 \
--do_lower_case \
--input_model ./SST-2_model &&
python3.7 -m onnxsim TinyBERT.onnx TinyBERT_sim.onnx

