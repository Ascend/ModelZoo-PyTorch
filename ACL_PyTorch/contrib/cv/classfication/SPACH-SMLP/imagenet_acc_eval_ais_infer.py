# Copyright 2020 Huawei Technologies Co., Ltd
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

# import pandas as pd
import numpy as np
import sys,os
from tqdm import tqdm

# Input params 

try:
	infer_result_dir = sys.argv[1]
except:
	infer_result_dir = "~/spach-smlp/ais_infer/2022_07_09-17_36_18"
try:
	n = int(sys.argv[2])
except:
	n = 50000

top_k = 5
acc_cnt = 0
acc_cnt_top5 = 0

# "/home/infname63/spach-smlp/ais_infer/2022_07_09-17_36_18/sample_id_0_output_0.bin"
for i in tqdm(range(n)):
	# infer_result_path = os.path.join(infer_result_dir, f"sample_id_{i}_output_0.bin")	
	# arr = np.fromfile(infer_result_path, dtype="float32")
	infer_result_path = os.path.join(infer_result_dir, f"sample_id_{i}_output_0.npy")	
	arr = np.load(infer_result_path)[0]

	infer_label = np.argmax(arr)
	arr_topk = np.argsort(arr)

	# print(i, ": ========")
	# print("Top 1", infer_label)
	# print("Top 5", arr_topk[-top_k:])
	# break
	true_label = i // 50
	if infer_label == true_label:
		acc_cnt += 1
	if true_label in arr_topk[-top_k:]:
		acc_cnt_top5 += 1
print(f"acc1:{acc_cnt / n:.4f}, acc5:{acc_cnt_top5 / n :.4f}")
# acc1:0.8174, acc5:0.9579