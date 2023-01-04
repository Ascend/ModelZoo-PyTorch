# Copyright 2022 Huawei Technologies Co., Ltd
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

import os
import sys
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

from ECAPA_TDNN.main import inference_embeddings_to_plt_hist_and_roc

if __name__ == "__main__":
    result_path = sys.argv[1]
    speakers_path = sys.argv[2]
    batch_size = 1
    total_nums = 4648
    embedding_holder = defaultdict(list)
    #样本总数为4648
    for i in tqdm(range(total_nums)):
        index = i+1
        speakers = torch.load(os.path.join(speakers_path, f"speakers{index}.pt"))

        bin_file_path = os.path.join(result_path, f"mels{index}_0.bin")
        batch = np.fromfile(bin_file_path, dtype='float32').reshape(-1, 192)
        h_tensor = torch.Tensor(batch)

        for h, s in zip(h_tensor.detach().cpu(), speakers):
            embedding_holder[s.item()].append(h.numpy())
    infer_hist, infer_roc, scores = inference_embeddings_to_plt_hist_and_roc(embedding_holder, 88600)
    tp, tn, roc_auc = scores
    print("roc_auc: ")
    print(roc_auc)