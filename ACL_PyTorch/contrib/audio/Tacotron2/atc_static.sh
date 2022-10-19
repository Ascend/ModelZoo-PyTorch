#!/bin/bash

SOC_VERSION=$1
batch_size=$2
seq_len=$3
cd ./output
atc --framework=5 --output=encoder_static  --input_format=ND  --soc_version=${SOC_VERSION} \
    --model=encoder.onnx \
    --input_shape="sequences:${batch_size},${seq_len};sequence_lengths:${batch_size}; \
    decoder_input:${batch_size},80;attention_hidden:${batch_size},1024;attention_cell:${batch_size},1024;decoder_hidden:${batch_size},1024;decoder_cell:${batch_size},1024; \
    attention_weights:${batch_size},${seq_len};attention_weights_cum:${batch_size},${seq_len};attention_context:${batch_size},512;mask:${batch_size},${seq_len};not_finished_input:${batch_size};mel_lengths_input:${batch_size}"  \
    --input_fp16_nodes="decoder_input;attention_hidden;attention_cell;decoder_hidden;decoder_cell;attention_weights;attention_weights_cum;attention_context" \
    --output_type="Gemm_118:0:FP16;Squeeze_87:0:FP16;Squeeze_88:0:FP16;Squeeze_115:0:FP16;Squeeze_116:0:FP16;Softmax_105:0:FP16;Add_109:0:FP16;Squeeze_108:0:FP16;PartitionedCall_Transpose_34_Transpose_38:0:FP16;MatMul_35:0:FP16;Unsqueeze_120:0:FP16;Unsqueeze_121:0:FP16" \
    --out_nodes="Gemm_118:0;Squeeze_87:0;Squeeze_88:0;Squeeze_115:0;Squeeze_116:0;Softmax_105:0;Add_109:0;Squeeze_108:0;PartitionedCall_Transpose_34_Transpose_38:0;MatMul_35:0;mask:0;Unsqueeze_120:0;Unsqueeze_121:0;Mul_128:0;Add_129:0"

out_names=$(python3 get_out_node.py decoder_sim_100.onnx)
out_type=$(python3 get_out_type.py decoder_sim_100.onnx)
atc --framework=5 --output=decoder_static  --input_format=ND  --soc_version=${SOC_VERSION} \
    --model=decoder_sim_100.onnx \
    --input_shape="decoder_input:${batch_size},80;attention_hidden:${batch_size},1024;attention_cell:${batch_size},1024;decoder_hidden:${batch_size},1024;decoder_cell:${batch_size},1024;attention_weights:${batch_size},${seq_len};attention_weights_cum:${batch_size},${seq_len};attention_context:${batch_size},512;
    memory:${batch_size},${seq_len},512;processed_memory:${batch_size},${seq_len},128;mask:${batch_size},${seq_len};gate_output_input:${batch_size},1,1;mel_output_input:${batch_size},80,1;not_finished_input:${batch_size};mel_lengths_input:${batch_size}"  \
    --input_fp16_nodes="decoder_input;attention_hidden;attention_cell;decoder_hidden;decoder_cell;attention_weights;attention_weights_cum;attention_context;memory_out;processed_memory_out;gate_output_input;mel_output_input"  \
    --out_nodes=$out_names \
    --output_type=$out_type

atc --framework=5 --output=postnet_static --input_format=ND  --soc_version=${SOC_VERSION} \
    --model=postnet.onnx \
    --input_shape="mel_outputs:${batch_size},80,2000" --input_fp16_nodes="mel_outputs"
