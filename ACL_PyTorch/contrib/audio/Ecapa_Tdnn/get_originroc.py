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

from collections import defaultdict

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from ECAPA_TDNN.main import ECAPA_TDNN, load_checkpoint, inference_embeddings_to_plt_hist_and_roc
from preprocess import get_dataloader
import sys

if __name__ == "__main__":
    device = torch.device("cpu")
    model = ECAPA_TDNN(1211, device).to(device)
    checkpoint = sys.argv[1]
    data_set = sys.argv[2]
    batch_size = 1
 
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=2e-5)

    model, optimizer, step = load_checkpoint(model, optimizer, checkpoint , rank='cpu')
    model.eval()
    acc_list = list()
    dataset_test, test_speakers = get_dataloader('vox1', 19, batch_size, data_set)
    embedding_holder = defaultdict(list)
    for mels, mel_length, speakers in tqdm(dataset_test):
        h_tensor, info_tensors = model(mels.to(device), infer=True)
        for h, s in zip(h_tensor.detach().cpu(), speakers):
            embedding_holder[s.item()].append(h.numpy())
    
    infer_hist, infer_roc, scores = inference_embeddings_to_plt_hist_and_roc(embedding_holder, step)
    tp, tn, roc_auc = scores
    print('origin roc_auc:')
    print(roc_auc)

