#!/bin/bash

# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
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
  echo "         bash convert_om.sh  xxx.air xx"

  exit 1
fi

input_onnx_path=$1
output_om_path=$2

atc \
	--model=$input_onnx_path \
  --framework=5 \
	--output=$output_om_path \
	--input_format=NCHW \
  --input_shape="input:1,3,800,1333" \
  --log=info \
	--soc_version=Ascend310 \
	--out_nodes="Concat_1232:0;Reshape_1238:0" \
	--insert_op_conf=fcos_aipp.cfg

