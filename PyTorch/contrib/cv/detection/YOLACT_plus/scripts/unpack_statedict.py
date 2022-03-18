# Copyright 2021 Huawei Technologies Co., Ltd
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
import torch
import sys, os

# Usage python scripts/unpack_statedict.py path_to_pth out_folder/
# Make sure to include that slash after your out folder, since I can't
# be arsed to do path concatenation so I'd rather type out this comment

print('Loading state dict...')
state = torch.load(sys.argv[1])

if not os.path.exists(sys.argv[2]):
	os.mkdir(sys.argv[2])

print('Saving stuff...')
for key, val in state.items():
	torch.save(val, sys.argv[2] + key)
