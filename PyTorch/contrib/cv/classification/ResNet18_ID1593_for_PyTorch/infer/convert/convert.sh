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
  echo "         bash $0 [INPUT_AIR_PATH] [OUTPUT_OM_PATH_NAME]"
  echo "Example: "
  echo "         bash convert.sh  xxx.onnx xx"

  exit 1
fi

input_air_path=$1
output_om_path=$2


echo "Input AIR file path: ${input_air_path}"
echo "Output OM file path: ${output_om_path}"

atc  --model=${input_air_path} \
--framework=5 \
--output=${output_om_path} \
--input_format=NCHW \
--soc_version=Ascend310 \
--output_type=FP32