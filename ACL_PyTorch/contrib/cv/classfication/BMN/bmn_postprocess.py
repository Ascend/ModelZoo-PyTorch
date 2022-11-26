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

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

import opts
sys.path.append(r"BMN-Boundary-Matching-Network")
sys.path.append(r"BMN-Boundary-Matching-Network/Evaluation")
from post_processing import BMN_post_processing
from dataset import VideoDataSet
from eval import run_evaluation, plot_metric

def gen_result_csv(opt):

    def get_result(idx):
        outputs = []
        for i in range(3):
            result_file = os.path.join(opt['result_dir'], '{:0>4d}_{}.bin'.format(int(idx), i))
            out = np.fromfile(result_file, dtype=np.float32)
            outputs.append(out)
        
        conf = torch.Tensor(outputs[0]).float().cpu().view(1, 2, 100, 100)
        start = torch.Tensor(outputs[1]).float().cpu().view(1, 100)
        end = torch.Tensor(outputs[-1]).float().cpu().view(1, 100)
        
        return conf, start, end
    
    if not os.path.exists("output/BMN_results"):
        os.makedirs("output/BMN_results")
    opt["mode"] = 'validation'
    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=False)
    tscale = opt["temporal_scale"]
    for idx, _ in tqdm(test_loader):
        video_name = test_loader.dataset.video_list[idx[0]]

        confidence_map, start, end = get_result(idx)

        start_scores = start[0].detach().cpu().numpy()
        end_scores = end[0].detach().cpu().numpy()
        clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
        reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()
        
        # Traverse the combination of the starting and ending cutoff points
        new_props = []
        for idx in range(tscale):
            for jdx in range(tscale):
                start_index = idx
                end_index = jdx + 1
                if start_index < end_index and end_index < tscale :
                    xmin = start_index / tscale
                    xmax = end_index / tscale
                    xmin_score = start_scores[start_index]
                    xmax_score = end_scores[end_index]
                    clr_score = clr_confidence[idx, jdx]
                    reg_score = reg_confidence[idx, jdx]
                    score = xmin_score * xmax_score * clr_score * reg_score
                    new_props.append([xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score])
        new_props = np.stack(new_props)

        col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_socre", "score"]
        new_df = pd.DataFrame(new_props, columns=col_name)
        new_df.to_csv("./output/BMN_results/" + video_name + ".csv", index=False)


def evaluation_proposal(opt):
    uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid = run_evaluation(
        ground_truth_filename=opt["ground_truth_file"],
        proposal_filename=opt["result_file"],
        max_avg_nr_proposals=100,
        tiou_thresholds=np.linspace(0.5, 0.95, 10),
        subset='validation')
    
    plot_metric(opt,uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid)
    
    print("AR@1 is \t", np.mean(uniform_recall_valid[:, 0]))
    print("AR@5 is \t", np.mean(uniform_recall_valid[:, 4]))
    print("AR@10 is \t", np.mean(uniform_recall_valid[:, 9]))
    print("AR@100 is \t", np.mean(uniform_recall_valid[:, -1]))


def post_processing(opt):
    gen_result_csv(opt)
    print("Post processing start")
    BMN_post_processing(opt)
    print("Post processing finished")
    evaluation_proposal(opt)
    

if __name__ == '__main__':
    option = opts.parse_opt()
    option = vars(option)
    post_processing(option)
