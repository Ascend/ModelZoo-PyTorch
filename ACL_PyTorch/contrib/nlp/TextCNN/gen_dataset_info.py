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

from glob import glob
import os, sys

dataset_path = sys.argv[1]
info_path = sys.argv[2]

bin_texts = glob(os.path.join(dataset_path, '*.bin'))
with open(info_path, 'w') as f:
	for index, texts in enumerate(bin_texts):
		content = str(index) + ' ' + str(texts) + '\n'
		f.write(content)
