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

import argparse
import os
import numpy as np
import torch.nn.functional as F
import mmcv
import torch
from mmcv import Config
from mmaction.datasets import build_dataset



def parse_args():

    parser = argparse.ArgumentParser(
        description='postprocess')  
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--data_path',
        default=None,
        help='the path of om result')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" ')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # Load eval_config from cfg
    eval_config = cfg.get('eval_config', {})
    eval_config = Config._merge_a_into_b(
            dict(metrics=args.eval), eval_config)
    dataset_type = cfg.data.test.type
    
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    result_path = args.data_path
    bin_path = os.listdir(result_path)[0]
    result_path = os.path.join(result_path, bin_path)
    bin_list = os.listdir(result_path)
    bin_list.sort(key= lambda x:int(x[:-6])) 
    outputs = []
    for i, bin in enumerate(bin_list):
        bin_path = os.path.join(result_path,bin_list[i])
        output = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 400)
        output = torch.from_numpy(output)
        output = F.softmax(output, dim=1).mean(dim=0)
        outputs.extend([output.numpy()])
         
    eval_res = dataset.evaluate(outputs, **eval_config)
    for name, val in eval_res.items():
        print(f'{name}: {val:.04f}')



if __name__ == '__main__':
    main()
