#!/bin/bash

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/pyACL/python/site-packages/acl:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

atc --framework=5 --model=./models/wav2vec2-base-960h.onnx --output=./models/wav2vec2-base-960h --input_format=ND --input_shape="input:1,-1" --dynamic_dims="10000;20000;30000;40000;50000;60000;70000;80000;90000;100000;110000;120000;130000;140000;150000;160000;170000;180000;190000;200000;210000;220000;230000;240000;250000;260000;270000;280000;290000;300000;310000;320000;330000;340000;350000;360000;370000;380000;390000;400000;410000;420000;430000;440000;450000;460000;470000;480000;490000;500000;510000;520000;530000;540000;550000;560000" --log=error --soc_version=Ascend310
