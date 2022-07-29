#!/bin/bash

# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if [ $# -ne 2 ]
then
  echo "Wrong parameter format."
  echo "Usage:"
  echo "         bash $0 [INPUT_ONNX_PATH] [OUTPUT_OM_PATH_NAME]"
  echo "Example: "
  echo "         bash convert_om.sh  xxx.onnx  xx"

  exit 1
fi

input_onnx_path=$1
output_om_path=$2

atc --framework=5 \
    --model="${input_onnx_path}" \
    --input_shape="image:1,3,224,224" \
    --output="${output_om_path}" \
    --input_format=NCHW \
    --output_type=FP32 \
    --op_select_implmode=high_precision \
    --soc_version=Ascend310