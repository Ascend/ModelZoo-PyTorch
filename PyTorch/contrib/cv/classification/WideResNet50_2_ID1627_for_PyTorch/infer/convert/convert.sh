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
# ============================================================================

if [ $# -ne 2 ]
then
  echo "Wrong parameter format."
  echo "Usage:"
  echo "         bash $0 [INPUT_ONNX_PATH] [OUTPUT_OM_PATH_NAME]"
  echo "Example: "
  echo "         bash convert.sh  ../data/model/wideresnet50.onnx  ../data/model/wideresnet50"

  exit 1
fi

model_path=$1
output_om_path=$2


echo "Input ONNX file path: ${input_air_path}"
echo "Output OM file path: ${output_om_path}"

atc \
--model=$model_path \
--framework=5 \
--output=$output_om_path \
--input_format=NCHW \
--input_shape="actual_input_1:1,3,304,304" \
--enable_small_channel=1 \
--disable_reuse_memory=1 \
--buffer_optimize=off_optimize \
--log=error \
--soc_version=Ascend310 \
--insert_op_conf=./aipp.config