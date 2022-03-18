# -*- coding: utf-8 -*-
# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import opts
from post_processing import BMN_post_processing
from dataset import VideoDataSet
import os
import numpy as np
import torch
import pandas as pd
from eval import evaluation_proposal


def gen_result_csv(opt):

    def get_result(idx):
        outputs = []
        for i in range(1, 4):
            out = np.fromfile('result/dumpOutput_device0/{:0>4d}_{}.bin'.format(int(idx), i), dtype=np.float32)
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
    for idx, input_data in test_loader:
        video_name = test_loader.dataset.video_list[idx[0]]

        # confidence_map, start, end = model(input_data)
        confidence_map, start, end = get_result(idx)

        start_scores = start[0].detach().cpu().numpy()
        end_scores = end[0].detach().cpu().numpy()
        clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
        reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()
        
        # 遍历起始分界点与结束分界点的组合
        new_props = []
        for idx in range(tscale):
            for jdx in range(tscale):
                start_index = idx
                end_index = jdx + 1
                if start_index < end_index and  end_index<tscale :
                    xmin = start_index / tscale
                    xmax = end_index / tscale
                    xmin_score = start_scores[start_index]
                    xmax_score = end_scores[end_index]
                    clr_score = clr_confidence[idx, jdx]
                    reg_score = reg_confidence[idx, jdx]
                    score = xmin_score * xmax_score * clr_score * reg_score
                    new_props.append([xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score])
        new_props = np.stack(new_props)
        #########################################################################

        col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_socre", "score"]
        new_df = pd.DataFrame(new_props, columns=col_name)
        new_df.to_csv("./output/BMN_results/" + video_name + ".csv", index=False)

def post_processing(opt):
    gen_result_csv(opt)
    print("Post processing start")
    BMN_post_processing(opt)
    print("Post processing finished")
    evaluation_proposal(opt)

if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    
    print('Current directory:', os.path.abspath('./'))
    post_processing(opt)
