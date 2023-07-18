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
from pathlib import Path
import platform
import sys
import numpy as np
from tqdm import tqdm

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, 'PaddleOCR')))

from ppocr.data import build_dataloader
import tools.program as program


def eval(config, logger):
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True, mode=644)
    batch_size = config['batch_size']
    valid_dataloader = build_dataloader(config, 'Eval', None, logger)
    max_iter = len(valid_dataloader) - 1 if platform.system() == "Windows" \
                    else len(valid_dataloader)

    logger.info('Processing, please wait a moment.')
    cnt = 0
    inputs = []
    for i, batch in tqdm(enumerate(valid_dataloader)):
        if i >= max_iter:
            break
        inputs.append(batch[0].numpy())
        if i % batch_size < batch_size - 1:
            continue
        np.concatenate(inputs, axis=0).tofile(str(save_dir / f'x-{cnt:0>2}.bin'))
        inputs = []
        cnt += 1
    logger.info(f'Done. Quantization data are saved in {save_dir}.')

def main():
    config, _, logger, _ = program.preprocess()
    config['Eval']['dataset']['data_dir'] = config['data_dir']
    label_file = Path(config['data_dir'])/'test_icdar2015_label.txt'
    config['Eval']['dataset']['label_file_list'] = [str(label_file)]
    eval(config, logger)


if __name__ == '__main__':
    main()
