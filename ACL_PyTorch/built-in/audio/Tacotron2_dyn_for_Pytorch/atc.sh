#!/bin/bash

SOC_VERSION=$1
batch_size=$2
max_seq_len=$3

echo "导出om：encoder"
atc --framework=5 --input_format=ND --soc_version=${SOC_VERSION} \
    --model=./output/encoder.onnx --output=./output/encoder_dyn \
    --input_shape_range="sequences:[${batch_size},1~${max_seq_len}];sequence_lengths:[${batch_size}]" \
    --log=error

echo "导出om：decoder"
atc --framework=5 --input_format=ND --soc_version=${SOC_VERSION} \
    --model=./output/decoder_iter.onnx --output=./output/decoder_iter_dyn \
    --input_shape_range="decoder_input:[${batch_size},80];attention_hidden:[${batch_size},1024];attention_cell:[${batch_size},1024];decoder_hidden:[${batch_size},1024];decoder_cell:[${batch_size},1024];attention_weights:[${batch_size},1~${max_seq_len}];attention_weights_cum:[${batch_size},1~${max_seq_len}];attention_context:[${batch_size},512];memory:[${batch_size},1~${max_seq_len},512];processed_memory:[${batch_size},1~${max_seq_len},128];mask:[${batch_size},1~${max_seq_len}]" \
    --log=error

echo "导出om：postnet"
atc --framework=5 --input_format=ND --soc_version=${SOC_VERSION} \
    --model=./output/postnet.onnx --output=./output/postnet_dyn \
    --input_shape_range="mel_outputs:[${batch_size},80,1~2000]" \
    --log=error
