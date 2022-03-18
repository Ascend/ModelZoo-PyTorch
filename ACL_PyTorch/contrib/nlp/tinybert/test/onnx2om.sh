#!/bin/bash

source ./test/env_npu.sh
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --input_format=ND --framework=5 --model=./TinyBERT_sim.onnx \
--input_shape="input_ids:1,64;segment_ids:1,64;input_mask:1,64" --output=TinyBERT --auto_tune_mode="RL,GA" \
--log=info --soc_version=Ascend310 --out_nodes='Gemm_423:0' --precision_mode=allow_mix_precision \
--modify_mixlist=./test/ops_info.json
