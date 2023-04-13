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

import os
import sys
import pickle
from pathlib import Path
import tqdm
import numpy as np
import paddle

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, 'PaddleOCR')))

from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
import tools.program as program


def postprocess(config):
    res_dir = Path(config['res_dir'])
    info_dir = Path(config['info_dir'])
    global_config = config['Global']
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)
    eval_class = build_metric(config['Metric'])
    maps_shape = (1, 1, 736, 1280) 
    for idx, out_file in enumerate(tqdm.tqdm(res_dir.iterdir())):
        if not out_file.name.endswith('_0.bin'):
            continue
        if out_file.name.startswith('padding_'):
            continue
        maps = np.fromfile(out_file, dtype=np.float32)
        maps = paddle.to_tensor(maps).reshape(maps_shape)
        preds = dict(maps=maps)
        info_file = info_dir/out_file.name.replace('_0.bin', '.pkl')
        with open(info_file, 'rb') as f:
            info = pickle.load(f)
        post_result = post_process_class(preds, info[1])
        eval_class(post_result, info)
    metric = eval_class.get_metric()
    for k, v in metric.items():
        print('{}:{}'.format(k, v))


if __name__ == '__main__':
    cfg, *_ = program.preprocess()
    postprocess(cfg)