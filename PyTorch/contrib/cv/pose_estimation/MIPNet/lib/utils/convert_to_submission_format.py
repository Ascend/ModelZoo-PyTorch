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

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import json

input_file = 'keypoints_test2017_results_epoch-1.json'
output_file = 'keypoints_test2017_results_epoch-1_pruned.json'

with open(input_file) as f:
	data = json.load(f)


new_data = []

keep_keys = ['image_id', 'category_id', 'keypoints', 'score']

for i, sample in enumerate(data):
	new_sample = {}

	for key in sample.keys():
		if key in keep_keys:
			new_sample[key] = sample[key]

	new_data.append(new_sample)
	print(i, len(data))

with open(output_file, 'w') as f:
	json.dump(new_data, f)