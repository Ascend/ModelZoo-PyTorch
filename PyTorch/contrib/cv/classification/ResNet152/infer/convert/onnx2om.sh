# /*
#  * Copyright (c) 2022. Huawei Technologies Co., Ltd
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  * you may not use this file except in compliance with the License.
#  * You may obtain a copy of the License at
# *
#  *     http://www.apache.org/licenses/LICENSE-2.0
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS,
#  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  * See the License for the specific language governing permissions and
#  * limitations under the License.
#  */

model_path=$1
framework=$2
output_model_name=$3

atc \
--model=$model_path \
--framework=$framework \
--output=$output_model_name \
--input_format=NCHW \
--input_shape="actual_input_1:1,3,304,304" \
--enable_small_channel=1 \
--disable_reuse_memory=1 \
--buffer_optimize=off_optimize \
--log=error \
--soc_version=Ascend310 \
--insert_op_conf=./aipp.config

