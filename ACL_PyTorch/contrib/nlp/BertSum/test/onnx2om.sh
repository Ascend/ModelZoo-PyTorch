#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd ../
atc --input_format=ND --framework=5 --model=./bertsum_13000_9_sim_bs1.onnx \
--input_shape="src:1,512;segs:1,512;clss:1,37;mask:1,512;mask_cls:1,37" --output=bertsum_13000_9_sim_bs1  \
--log=info --soc_version=Ascend310 --precision_mode=allow_mix_precision  \
--modify_mixlist=ops_info.json
