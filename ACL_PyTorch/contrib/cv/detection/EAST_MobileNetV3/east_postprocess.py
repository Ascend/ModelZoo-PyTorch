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


import pickle
from pathlib import Path

import numpy as np
import paddle

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

    geo_shape = (1, 8, 176, 320)
    score_shape = (1, 1, 176, 320)

    for out_file0 in res_dir.iterdir():
        if not out_file0.name.endswith('_0.bin'):
            continue
        if out_file0.name.startswith('padding_'):
            continue

        f_geo = np.fromfile(out_file0, dtype=np.float32)
        f_geo = paddle.to_tensor(f_geo).reshape(geo_shape)
        out_file1 = str(out_file0).replace('_0.bin', '_1.bin')
        f_score = np.fromfile(out_file1, dtype=np.float32)
        f_score = paddle.to_tensor(f_score).reshape(score_shape)
        preds = dict(f_geo=f_geo, f_score=f_score)

        info_file = info_dir/out_file0.name.replace('_0.bin', '.pkl')
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
