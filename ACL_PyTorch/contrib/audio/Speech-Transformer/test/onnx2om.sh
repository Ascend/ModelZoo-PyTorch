# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --framework=5 --model=encoder.onnx --output=encoder --input_format=NCHW --input_shape="padded_input:1,512,320;non_pad_mask:1,512,1;slf_attn_mask:1,512,512" --log=error --soc_version=Ascend310
atc --framework=5 --model=decoder.onnx --output=decoder --input_format=NCHW --input_shape="ys_in:1,128; encoder_outputs:1,512,512;non_pad_mask:1,128,1;slf_attn_mask:1,128,128" --log=error --soc_version=Ascend310
atc --framework=5 --model=tgt_word_prj.onnx --output=tgt_word_prj --input_format=NCHW --input_shape="input:1,512" --log=error --soc_version=Ascend310