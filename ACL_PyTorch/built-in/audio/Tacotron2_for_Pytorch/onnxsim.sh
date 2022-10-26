input_model=$1
output_model=$2
batch_size=$3
seq_len=$4
python3 -m onnxsim --input-shape "decoder_input:${batch_size},80" "attention_hidden:${batch_size},1024" "attention_cell:${batch_size},1024" "decoder_hidden:${batch_size},1024" "decoder_cell:${batch_size},1024" "attention_weights:${batch_size},${seq_len}" "attention_weights_cum:${batch_size},${seq_len}" "attention_context:${batch_size},512" \
    "memory:${batch_size},${seq_len},512" "processed_memory:${batch_size},${seq_len},128" "mask:${batch_size},${seq_len}" "gate_output_input:${batch_size},1,1" "mel_output_input:${batch_size},80,1" "not_finished_input:${batch_size}" "mel_lengths_input:${batch_size}" --dynamic-input-shape $input_model $output_model
