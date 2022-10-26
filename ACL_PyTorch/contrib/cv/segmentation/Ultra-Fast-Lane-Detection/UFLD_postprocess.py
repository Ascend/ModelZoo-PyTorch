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
import json
import tqdm
from pathlib import Path

import numpy as np
import torch

from utils.dist_utils import dist_print, get_rank
from evaluation.tusimple.lane import LaneEval
from evaluation.eval_wrapper import generate_tusimple_lines, combine_tusimple_test


def run_test_tusimple(work_dir, exp_name, griding_num=100, use_aux=True, 
                      distributed=False, batch_size=1):
    
    work_dir = Path(work_dir)
    output_path = work_dir/f'{exp_name}.{get_rank()}.txt'
    fp = open(output_path, 'w')
    for res_path in tqdm.tqdm(Path(work_dir).iterdir()):
        if not res_path.stem.startswith('clips'):
            continue
        tmp_dict = {}
        if res_path.suffix == '.txt':
            out = np.loadtxt(res_path, dtype=np.float32)
            tmp_dict['raw_file'] = res_path.name.replace('_0.txt','.jpg').replace('-', '/')
        elif res_path.suffix == '.bin':
            out = np.fromfile(res_path, dtype=np.float16).astype(np.float32)
            tmp_dict['raw_file'] = res_path.name.replace('_0.bin','.jpg').replace('-', '/')
        elif res_path.suffix == '.npy':
            out = np.load(res_path).astype(np.float32)
            tmp_dict['raw_file'] = res_path.name.replace('_0.npy','.jpg').replace('-', '/')
        else:
            raise Exception('Unknown file type.')
        out = torch.from_numpy(out).reshape(1, griding_num+1, 56, 4)
        if len(out)==2 and use_aux:
          out = out[0]
        tmp_dict['lanes'] = generate_tusimple_lines(out[0], None, griding_num = 100)
        tmp_dict['h_samples'] = a = list(range(160, 711, 10))
        tmp_dict['run_time'] = 10
        json_str = json.dumps(tmp_dict)

        fp.write(json_str + '\n')
    fp.close()


def eval_lane(work_dir, label_path, griding_num=100, use_aux=True, distributed=False):
    exp_name = 'tusimple_eval_temp'
    run_test_tusimple(work_dir, exp_name, griding_num, use_aux, distributed)
    combine_tusimple_test(work_dir, exp_name)
    combined_path = Path(work_dir)/f'{exp_name}.{get_rank()}.txt'
    res = LaneEval.bench_one_submit(combined_path, label_path)
    res = json.loads(res)
    for r in res:
        dist_print(r['name'], r['value'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser( 'postprocess and compute accuracy.')
    parser.add_argument('--result-path', type=str, help='path to infer results.')
    parser.add_argument('--label-path', type=str, help='path to label file.')
    args = parser.parse_args()

    eval_lane(args.result_path, args.label_path)
