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


from glob import glob
import os.path as osp

import numpy as np
import paddle
from tqdm import tqdm

from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
import tools.program as program


def model_eval(config, logger):
    output_dir = config['output_dir']
    info_dir = config['info_dir']
    post_process_class = build_post_process(
                            config['PostProcess'], config['Global'])
    eval_class = build_metric(config['Metric'])
    
    output_list = glob(osp.join('om_outputs', '**', 'image-*_0.bin'), 
                       recursive=True)
    for output_bin in tqdm(output_list, desc='model eval'):
        output = np.fromfile(output_bin, dtype=np.float16).astype(np.float32)
        preds = {'maps': paddle.to_tensor(output.reshape(1, 1, 736, 1280))}
        idx = output_bin.rsplit('-', 1)[-1].split('_', 1)[0]
        info = np.load(osp.join(info_dir, f'info-{idx}.npz'), 
                       allow_pickle=True)['arr_0'][()]
        batch_numpy = [info[k] for k in sorted(info.keys())]
        post_result = post_process_class(preds, batch_numpy[1])
        eval_class(post_result, batch_numpy)

    logger.info('↓↓↓↓↓↓↓↓↓↓↓ Metrics ↓↓↓↓↓↓↓↓↓↓↓')
    metric = eval_class.get_metric()
    for k, v in metric.items():
        logger.info('{} = {}'.format(k, v))


def main():
    config, _, logger, _ = program.preprocess()
    model_eval(config, logger)


if __name__ == '__main__':
    main()
