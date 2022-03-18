#!/bin/bash

if [ $# -ne 3 ]
then
  echo "Wrong parameter format."
  echo "Usage:"
  echo "         bash $0 [INPUT_AIR_PATH] [AIPP_PATH] [OUTPUT_OM_PATH_NAME]"
  echo "Example: "
  echo "         bash convert_om.sh  xxx.air ./aipp.cfg xx"

  exit 1
fi

input_onnx_path=$1
aipp_cfg_file=$2
output_om_path=$3

export install_path=/usr/local/Ascend/

export ASCEND_ATC_PATH=${install_path}/atc
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/latest/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg
export ASCEND_OPP_PATH=${install_path}/opp
echo "Input AIR file path: ${input_onnx_path}"
echo "Output OM file path: ${output_om_path}"

atc --input_format=NCHW \
    --framework=5 \
    --model="${input_onnx_path}" \
    --input_shape="image:1,3,240,240"  \
    --output="${output_om_path}" \
    --insert_op_conf="${aipp_cfg_file}" \
    --enable_small_channel=1 \
    --log=error \
    --soc_version=Ascend310 \
    --op_select_implmode=high_precision \
    --output_type=FP32
