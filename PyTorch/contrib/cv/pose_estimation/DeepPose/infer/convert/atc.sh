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

onnx_path=$1
om_path=$2
echo "Input ONNX file path: ${onnx_path}"
echo "Output OM file path: ${om_path}"

atc --model="${onnx_path}" \
    --framework=5 \
    --output="${om_path}" \
    --soc_version=Ascend310 \
    --input_format=NCHW \
    --input_shape="actual_input_1:1,3,256,192" \
    --log=error \
    --output_type=FP32 