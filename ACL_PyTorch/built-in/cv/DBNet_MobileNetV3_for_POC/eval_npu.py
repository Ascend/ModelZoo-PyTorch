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


from math import ceil
import os
from pathlib import Path
import pickle
import platform
import sys

import numpy as np
import paddle
from tqdm import tqdm

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, 'PaddleOCR')))

from ppocr.data import build_dataloader
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
import tools.program as program
from ais_bench.infer.interface import InferSession


class OmInferencer:
    def __init__(self, om_path=None, device_id=0):
        self.session = InferSession(device_id, om_path)
        input_shape = self.session.get_inputs()[0].shape
        self.batch_size = input_shape[0]
        self.with_aipp = (len(input_shape) == 4 and input_shape[-1] == 3)

    def infer(self, inputs):
        num_inputs = len(inputs)
        assert num_inputs <= self.batch_size
        num_pad = self.batch_size - len(inputs)
        inputs_ = inputs + [inputs[-1]] * num_pad
        inputs_ = np.concatenate(inputs_, axis=0)
        outputs = self.session.infer([inputs_])
        return np.split(outputs[0][:num_inputs, ...], num_inputs, axis=0)

    def close(self):
        self.session.sumary()
        self.session.finalize()


def eval(config, logger):
    device_id = int(config['device_id']) if 'device_id' in config else 0
    om_inferencer = OmInferencer(om_path=config['om_path'], device_id=device_id)
    batch_size = om_inferencer.batch_size
    if om_inferencer.with_aipp:
        transforms_config = config['Eval']['dataset']['transforms' ]
        assert 'NormalizeImage' in transforms_config[3]
        # remove NormalizeImage because AIPP can do same thing
        transforms_config.pop(3)
        assert 'ToCHWImage' in transforms_config[3]
        # remove ToCHWImage because AIPP require HWC
        transforms_config.pop(3)

    # valid_dataloader = build_dataloader(config, 'Eval', device, logger)
    valid_dataloader = build_dataloader(config, 'Eval', None, logger)
    post_process_class = build_post_process(config['PostProcess'],
                                            config['Global'])
    eval_class = build_metric(config['Metric'])
    max_iter = len(valid_dataloader)
    if platform.system() == "Windows":
        max_iter -= 1

    inputs, infos = [], []
    total_size = len(valid_dataloader)
    num_batchs = ceil(len(valid_dataloader) / batch_size)
    logger.info(f'Total: {total_size} images  {num_batchs} batchs.')
    pbar = tqdm(total=num_batchs)
    for i, batch in enumerate(valid_dataloader):
        if i >= max_iter:
            break
        info = [item.numpy() for item in batch]
        image = batch[0].numpy()
        if om_inferencer.with_aipp:
            image = image.astype(np.uint8)

        inputs.append(image)
        infos.append(info)
        if len(inputs) < batch_size:
            continue

        outputs = om_inferencer.infer(inputs)
        for out, info_ in zip(outputs, infos):
            preds = {'maps': paddle.to_tensor(out.astype(np.float32))}
            post_result = post_process_class(preds, info_[1])
            eval_class(post_result, info_)
        inputs, infos = [], []
        pbar.update(1)

    if inputs:
        outputs = om_inferencer.infer(inputs)
        for out, info_ in zip(outputs, infos):
            preds = {'maps': paddle.to_tensor(out.astype(np.float32))}
            post_result = post_process_class(preds, info_[1])
            eval_class(post_result, info_)
        pbar.update(1)        

    pbar.close()
    om_inferencer.close()
    logger.info('↓↓↓↓↓↓↓↓↓↓↓ Metrics ↓↓↓↓↓↓↓↓↓↓↓')
    metric = eval_class.get_metric()
    for k, v in metric.items():
        logger.info('{} = {}'.format(k, v))


def main():
    config, _, logger, _ = program.preprocess()
    config['Eval']['dataset']['data_dir'] = config['data_dir']
    label_file = Path(config['data_dir'])/'test_icdar2015_label.txt'
    config['Eval']['dataset']['label_file_list'] = [str(label_file)]
    eval(config, logger)


if __name__ == '__main__':
    main()
