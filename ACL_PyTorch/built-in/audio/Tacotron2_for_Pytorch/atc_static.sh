#!/bin/bash

export install_path="/usr/local/Ascend/ascend-toolkit/latest"
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

export ASCEND_SLOG_PRINT_TO_STDOUT=1

seq_len=128

SOC_VERSION=$1
batch_size=$2

${install_path}/atc/bin/atc --framework=5 --output=./output/encoder_static  --input_format=ND  --soc_version=${SOC_VERSION} \
    --model=./output/encoder_modify.onnx \
    --input_shape="sequences:${batch_size},${seq_len};sequence_lengths:${batch_size}"  \
    --out_nodes="PartitionedCall_Transpose_34_Transpose_21:0;MatMul_37:0;Mul_36:0" \
    --output_type="PartitionedCall_Transpose_34_Transpose_21:0:FP16;MatMul_37:0:FP16"

${install_path}/atc/bin/atc --framework=5 --output=./output/decoder_static  --input_format=ND  --soc_version=${SOC_VERSION} \
    --model=./output/decoder_iter_modify.onnx \
    --input_shape="decoder_input:${batch_size},80;attention_hidden:${batch_size},1024;attention_cell:${batch_size},1024;decoder_hidden:${batch_size},1024;decoder_cell:${batch_size},1024;attention_weights:${batch_size},${seq_len};attention_weights_cum:${batch_size},${seq_len};attention_context:${batch_size},512;memory:${batch_size},${seq_len},512;processed_memory:${batch_size},${seq_len},128;mask:${batch_size},${seq_len};random1:256;random2:256"  \
    --input_fp16_nodes="decoder_input;attention_hidden;attention_cell;decoder_hidden;decoder_cell;attention_weights;attention_weights_cum;attention_context;memory;processed_memory;random1;random2"  \
    --out_nodes="Gemm_82:0;Gemm_83:0;Squeeze_51:0;Squeeze_52:0;Squeeze_79:0;Squeeze_80:0;Softmax_69:0;Add_73:0;Squeeze_72:0" \
    --output_type="Gemm_82:0:FP16;Gemm_83:0:FP16;Squeeze_51:0:FP16;Squeeze_52:0:FP16;Squeeze_79:0:FP16;Squeeze_80:0:FP16;Softmax_69:0:FP16;Add_73:0:FP16;Squeeze_72:0:FP16"


${install_path}/atc/bin/atc --framework=5 --output=./output/postnet_static --input_format=ND  --soc_version=${SOC_VERSION} \
    --model=./output/postnet.onnx \
    --input_shape="mel_outputs:${batch_size},80,2000" --input_fp16_nodes="mel_outputs"
