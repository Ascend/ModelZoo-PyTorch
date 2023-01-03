# Copyright 2021 Huawei Technologies Co., Ltd
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
import argparse
import torch
import os
import numpy as np
from tqdm import tqdm


def postProcesss(head_result_path, tail_result_path, data_head, data_tail):

    bin_head_list = os.listdir(head_result_path)
    bin_head_list.sort(key=lambda x: int(x.split('-')[0][3:]))

    bin_tail_list = os.listdir(tail_result_path)
    bin_tail_list.sort(key=lambda x: int(x.split('-')[0][3:]))
    head_ite_list = os.listdir(data_head+'/post')
    tail_ite_list = os.listdir(data_tail+'/post')
    head_pos_list = os.listdir(data_head+'/possamp')
    tail_pos_list = os.listdir(data_head + '/possamp')
    head_ite_list.sort(key=lambda x: int(x.split('-')[0][3:]))
    tail_ite_list.sort(key=lambda x: int(x.split('-')[0][3:]))
    head_pos_list.sort(key=lambda x: int(x.split('-')[0][3:]))
    tail_pos_list.sort(key=lambda x: int(x.split('-')[0][3:]))

    logs = []
    for i in tqdm(range(len(bin_head_list)), desc="Postprocessing head data..."):
        bin_path = os.path.join(head_result_path, bin_head_list[i])
        score = np.load(bin_path)
        score = torch.from_numpy(score)
        ite_path = os.path.join(data_head+'/post', head_ite_list[i])
        filter_bias = np.loadtxt(ite_path)
        filter_bias = torch.from_numpy(filter_bias)
        pos_path = os.path.join(data_head + '/possamp', head_pos_list[i])
        positive_sample = np.loadtxt(pos_path)
        positive_sample = positive_sample.reshape(-1, 3)
        score += filter_bias
        score = torch.reshape(score, (-1, 14541))
        # Explicitly sort all the entities to ensure that there is no test exposure bias
        argsort = torch.argsort(score, dim=1, descending=True)
        positive_arg = positive_sample[:, 0]

        for i in range(len(score)):
            # Notice that argsort is not ranking
            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
            assert ranking.size(0) == 1
            # ranking + 1 is the true ranking used in evaluation metrics
            ranking = 1 + ranking.item()
            logs.append({
                'MRR': 1.0 / ranking,
                'MR': float(ranking),
                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                'HITS@10': 1.0 if ranking <= 10 else 0.0,
            })
    for i in tqdm(range(len(bin_tail_list)), desc="Postprocessing tail data..."):
        bin_path = os.path.join(tail_result_path, bin_tail_list[i])
        score = np.load(bin_path)
        score = torch.from_numpy(score)
        ite_path = os.path.join(data_tail + '/post', tail_ite_list[i])
        filter_bias = np.loadtxt(ite_path)
        filter_bias = torch.from_numpy(filter_bias)
        pos_path = os.path.join(data_tail + '/possamp', tail_pos_list[i])
        positive_sample = np.loadtxt(pos_path)
        positive_sample = positive_sample.reshape(-1, 3)
        score += filter_bias
        score = torch.reshape(score, (-1, 14541))

        # Explicitly sort all the entities to ensure that there is no test exposure bias
        argsort = torch.argsort(score, dim=1, descending=True)
        positive_arg = positive_sample[:, 2]

        for i in range(len(score)):
            # Notice that argsort is not ranking
            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
            assert ranking.size(0) == 1
            # ranking + 1 is the true ranking used in evaluation metrics
            ranking = 1 + ranking.item()
            logs.append({
                'MRR': 1.0 / ranking,
                'MR': float(ranking),
                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                'HITS@10': 1.0 if ranking <= 10 else 0.0,
            })

    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='postprocess of rotate')

    parser.add_argument(
        '--head_result_path', default=r'RotatEout/bs1/head')
    parser.add_argument(
        '--tail_result_path', default=r'RotatEout/bs1/tail')
    parser.add_argument(
        '--data_head', default=r'bin/head')
    parser.add_argument(
        '--data_tail', default=r'bin/tail')
    opt = parser.parse_args()
    metrics = postProcesss(opt.head_result_path,
                           opt.tail_result_path,
                           opt.data_head,
                           opt.data_tail)
    print(metrics)
