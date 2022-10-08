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


python Tdnn_pyacl_infer.py --model_path=tdnn.om --device_id=0 --cpu_run=True --sync_infer=True --workspace=10 --input_info_file_path=mini_librispeech_test.info --input_dtypes=float32 --infer_res_save_path=result --res_save_type=bin
