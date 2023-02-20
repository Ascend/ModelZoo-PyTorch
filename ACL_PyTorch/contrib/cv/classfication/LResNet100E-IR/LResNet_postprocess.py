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
# limitations under the License

import os
import sys
sys.path.append("./LResNet")
import argparse
import numpy as np
from tqdm import tqdm
import torch
from verifacation import evaluate



def l2_norm(input,axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def evaluation_om(result_dir, target_list_path):
    issame = np.load(target_list_path)
    
    for root, dirs, files in os.walk(result_dir):
        results_path = os.path.join(result_dir, dirs[-1])
        break

    result_ids = next(os.walk(results_path))
    n = len(result_ids[-1])//2
    embeddings = np.zeros([n, 512], dtype=np.float32)
    for idx in tqdm(range(n)):
        result_path = os.path.join(results_path, f'{idx}_0.txt')
        result_flip_path = os.path.join(results_path, f'{idx}_flip_0.txt')
        emb = np.loadtxt(result_path, dtype=np.float32) + np.loadtxt(result_flip_path, dtype=np.float32)
        embeddings[idx:idx+1] = l2_norm(torch.tensor(emb).unsqueeze(0)).detach().numpy()

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds=10)
    print('*'*50)
    print('accuracy: {}'.format(accuracy.mean()))
    print('best_thresholds: {}'.format(best_thresholds.mean()))
    print('*'*50)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=str, default="./result/dumpOutput_device0")
    parser.add_argument('--data_path', type=str, default="./data/lfw_list.npy")
    args = parser.parse_args()
    evaluation_om(args.result, args.data_path)
