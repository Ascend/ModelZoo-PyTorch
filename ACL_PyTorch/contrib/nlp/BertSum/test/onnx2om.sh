#!/bin/bash
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_GLOBAL_LOG_LEVEL=3
cd ../
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --input_format=ND --framework=5 --model=./bertsum_13000_9_sim_bs1.onnx \
--input_shape="src:1,512;segs:1,512;clss:1,37;mask:1,512;mask_cls:1,37" --output=bertsum_13000_9_sim_bs1  \
--log=info --soc_version=Ascend310 --precision_mode=allow_mix_precision  \
--modify_mixlist=ops_info.json
