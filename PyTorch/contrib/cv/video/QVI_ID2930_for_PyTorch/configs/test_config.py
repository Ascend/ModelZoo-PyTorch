# Copyright 2020 Huawei Technologies Co., Ltd
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
testset_root = './datasets/example'
test_size = (854, 480)
test_crop_size = (854, 480)

mean = [0.429, 0.431, 0.397]
std = [1, 1, 1]

inter_frames = 3

model = 'QVI'
pwc_path = './utils/pwc-checkpoint.pt'

store_path = 'demo_out/example/'
checkpoint = 'qvi_release/model.pt'
