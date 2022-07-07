#!/bin/bash

# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 3.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-3.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

input_onnx_path=$1
output_om_path=$2

echo "Input ONNX file path: ${input_onnx_path}"
echo "Output OM file path: ${output_om_path}"

atc --framework=5 \
    --model="${input_onnx_path}" \
    --input_shape="actual_input_1:1,3,224,224" \
    --output="${output_om_path}" \
    --input_format=NCHW \
    --output_type=FP32 \
    --soc_version=Ascend310
