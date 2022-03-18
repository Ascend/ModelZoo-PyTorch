#!/bin/bash
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
#export DUMP_GE_GRAPH=2

atc --model=$1 --framework=5 --input_format=ND --input_shape="feats:1,-1,23" --dynamic_dims='200;300;400;500;600;700;800;900;1000;1100;1200;1300;1400;1500;1600;1700;1800' --output=$2 --soc_version=Ascend310 --log=info