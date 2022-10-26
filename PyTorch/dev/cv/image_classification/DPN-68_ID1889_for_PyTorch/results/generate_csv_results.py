#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
import numpy as np
import pandas as pd
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))


results = {
    'results-imagenet.csv': [
        'results-imagenet-real.csv',
        'results-imagenetv2-matched-frequency.csv',
        'results-sketch.csv'
    ],
    'results-imagenet-a-clean.csv': [
        'results-imagenet-a.csv',
    ],
    'results-imagenet-r-clean.csv': [
        'results-imagenet-r.csv',
    ],
}


def diff(base_df, test_csv):
    base_models = base_df['model'].values
    test_df = pd.read_csv(test_csv)
    test_models  = test_df['model'].values

    rank_diff = np.zeros_like(test_models, dtype='object')
    top1_diff = np.zeros_like(test_models, dtype='object')
    top5_diff = np.zeros_like(test_models, dtype='object')
    
    for rank, model in enumerate(test_models):
        if model in base_models:            
            base_rank = int(np.where(base_models == model)[0])
            top1_d = test_df['top1'][rank] - base_df['top1'][base_rank]
            top5_d = test_df['top5'][rank] - base_df['top5'][base_rank]
            
            # rank_diff
            if rank == base_rank:
                rank_diff[rank] = f'0'
            elif rank > base_rank:
                rank_diff[rank] = f'-{rank - base_rank}'
            else:
                rank_diff[rank] = f'+{base_rank - rank}'
                
            # top1_diff
            if top1_d >= .0:
                top1_diff[rank] = f'+{top1_d:.3f}'
            else:
                top1_diff[rank] = f'-{abs(top1_d):.3f}'
            
            # top5_diff
            if top5_d >= .0:
                top5_diff[rank] = f'+{top5_d:.3f}'
            else:
                top5_diff[rank] = f'-{abs(top5_d):.3f}'
                
        else: 
            rank_diff[rank] = ''
            top1_diff[rank] = ''
            top5_diff[rank] = ''

    test_df['top1_diff'] = top1_diff
    test_df['top5_diff'] = top5_diff
    test_df['rank_diff'] = rank_diff

    test_df['param_count'] = test_df['param_count'].map('{:,.2f}'.format)
    test_df.sort_values('top1', ascending=False, inplace=True)
    test_df.to_csv(test_csv, index=False, float_format='%.3f')


for base_results, test_results in results.items():
    base_df = pd.read_csv(base_results)
    base_df.sort_values('top1', ascending=False, inplace=True)
    for test_csv in test_results:
        diff(base_df, test_csv)
    base_df['param_count'] = base_df['param_count'].map('{:,.2f}'.format)
    base_df.to_csv(base_results, index=False, float_format='%.3f')
