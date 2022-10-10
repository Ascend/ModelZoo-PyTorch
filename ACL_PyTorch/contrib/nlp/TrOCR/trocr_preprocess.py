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

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, 'slim')))
sys.path.append(os.path.abspath(os.path.join(__dir__, 'fairseq')))

import argparse
import numpy as np
import torch
import fairseq
from tqdm import tqdm
from unilm.trocr import task

def run_preprocess(args):
    model_path = args.model_path
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [model_path],
            arg_overrides={"data_type":"STR", "beam": 10, "task": "text_recognition", "data": args.datasets_path, "fp16": False})

    task.load_dataset(cfg.dataset.gen_subset, task_cfg=cfg.task)
    
    with tqdm(total=2915) as pbar:
        k = 0
        save_path = args.pre_data_save_path
        for i in task.datasets['test']:
            pbar.update(1)
            tfm_img = np.array(i['tfm_img'])
            label_ids = np.array(i['label_ids'])
            tfm_img.tofile(os.path.join(save_path + "/tfm_img_" + str(k) + '.bin'))
            k += 1
    print("{} bin files generated successfully.".format(k))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./trocr-small-handwritten.pt')
    parser.add_argument('--datasets_path', default='./IAM/')
    parser.add_argument('--pre_data_save_path', default='./pre_data')
    args = parser.parse_args()
    if not os.path.exists(args.pre_data_save_path):
        os.makedirs(args.pre_data_save_path)

    run_preprocess(args)

