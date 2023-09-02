# Copyright 2023 Huawei Technologies Co., Ltd
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

from espnet_onnx.export import ASRModelExport
m = ASRModelExport()
m.set_export_config(max_seq_len=2048, ctc_weight=0.3)
m.export_from_zip(
    'asr_train_asr_conformer3_raw_char_batch_bins4000000_accum_grad4_sp_valid.acc.ave.zip', 
    tag_name='asr_train_asr_qkv', 
    ctc_weight=0.3, 
    lm_weight=0.3
)
