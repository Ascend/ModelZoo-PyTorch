#!/bin/bash

SOC_VERSION=$1
batch_size=$2
seq_len=$3
cd ./output
atc --framework=5 --output=encoder_static  --input_format=ND  --soc_version=${SOC_VERSION} \
    --model=encoder.onnx \
    --input_shape="sequences:${batch_size},${seq_len};sequence_lengths:${batch_size}; \
    decoder_input:${batch_size},80;attention_hidden:${batch_size},1024;attention_cell:${batch_size},1024;decoder_hidden:${batch_size},1024;decoder_cell:${batch_size},1024; \
    attention_weights:${batch_size},${seq_len};attention_weights_cum:${batch_size},${seq_len};attention_context:${batch_size},512;mask.1:${batch_size},${seq_len};not_finished_input:${batch_size};mel_lengths_input:${batch_size}"

atc --framework=5 --output=decoder_static  --input_format=ND  --soc_version=${SOC_VERSION} \
    --model=decoder_sim_100.onnx \
    --input_shape="decoder_input:${batch_size},80;attention_hidden:${batch_size},1024;attention_cell:${batch_size},1024;decoder_hidden:${batch_size},1024;decoder_cell:${batch_size},1024;attention_weights:${batch_size},${seq_len};attention_weights_cum:${batch_size},${seq_len};attention_context:${batch_size},512;
    memory:${batch_size},${seq_len},512;processed_memory:${batch_size},${seq_len},128;mask:${batch_size},${seq_len};gate_output_input:${batch_size},1,1;mel_output_input:${batch_size},80,1;not_finished_input:${batch_size};mel_lengths_input:${batch_size}"

atc --framework=5 --output=postnet_static --input_format=ND  --soc_version=${SOC_VERSION} \
    --model=postnet.onnx \
    --input_shape="mel_outputs:${batch_size},80,2000"
