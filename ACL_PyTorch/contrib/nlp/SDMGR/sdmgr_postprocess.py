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
# limitations under the License.


import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
from mmcv import Config

from mmocr.datasets import build_dataset


def main():
    parser = argparse.ArgumentParser(
                        description='postprocess.')
    parser.add_argument('--config', type=str, required=True, 
                        help='Test config file path.')
    parser.add_argument('--res-dir', type=str, required=True, 
                        help='a directory to save binary files.')
    args = parser.parse_args()
    postprocess(args.config, args.res_dir)
    


def postprocess(config_path, res_dir):

    cfg = Config.fromfile(config_path)
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    num_data = len(dataset)
    res_dir = Path(res_dir)

    results = []
    for i in tqdm(range(num_data)):
        data = dataset[i]
        img_name = data['img_metas'].data['ori_filename']
        img_stem = Path(img_name.replace('/', '-')).stem

        res_file1 = res_dir / f"{img_stem}_0.npy"
        res_file2 = res_dir / f"{img_stem}_1.npy"
        nodes = torch.from_numpy(np.load(res_file1))
        edges = torch.from_numpy(np.load(res_file2))

        result = [dict(
            img_metas=data['img_metas'].data,
            nodes=nodes, edges=edges
        )]
        results.extend(result)

    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in ['interval', 'tmpdir', 'start', 
                'gpu_collect', 'save_best', 'rule']:
        eval_kwargs.pop(key, None)
    eval_kwargs['metric'] = 'macro_f1'
    metric = dataset.evaluate(results, **eval_kwargs)
    print(metric)


if __name__ == '__main__':
    main()
