#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --input_format=ND --framework=5 --model=./TinyBERT_sim_bs$1.onnx \
--input_shape="input_ids:$1,64;segment_ids:$1,64;input_mask:$1,64" --output=TinyBERT_bs$1  --log=info \
--soc_version=Ascend$2 --out_nodes='Gemm_423:0' --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance
