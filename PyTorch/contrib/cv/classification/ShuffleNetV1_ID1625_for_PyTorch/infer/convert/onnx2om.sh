#!/usr/bin/env bash

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
#--input_format=NCHW \
#--input_shape="actual_input_1:1,3,224,224" \
model=$1
atc --model=$model \
    --framework=5 \
    --output=../data/model/ShuffleNetV1 \
    --input_format=NCHW \
    --input_shape="actual_input_1:1,3,224,224" \
    --soc_version=Ascend310 \
    --precision_mode=allow_mix_precision \
    --op_select_implmode=high_precision