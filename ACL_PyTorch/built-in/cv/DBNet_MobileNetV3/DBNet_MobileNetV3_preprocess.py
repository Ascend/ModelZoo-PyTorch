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
import stat
import pickle
import platform
from pathlib import Path
import tqdm
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, 'PaddleOCR')))

from ppocr.data import build_dataloader
import tools.program as program


def preprocess(config, device, logger):
    bin_dir = Path(config['bin_dir'])
    info_dir = Path(config['info_dir'])
    bin_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)
    transforms_config = config['Eval']['dataset']['transforms' ]
    assert 'NormalizeImage' in transforms_config[3]
    # remove NormalizeImage because AIPP can do same thing
    transforms_config.pop(3)
    assert 'ToCHWImage' in transforms_config[3]
    # remove ToCHWImage because AIPP require HWC
    transforms_config.pop(3)
    valid_dataloader = build_dataloader(config, 'Eval', device, logger)
    max_iter = len(valid_dataloader)
    if platform.system() == "Windows":
        max_iter -= 1
    for idx, batch in enumerate(tqdm.tqdm(valid_dataloader)):
        if idx >= max_iter:
            break
        image = batch[0]
        bin_path = bin_dir/f'image-{idx:0>3}.bin'
        image.numpy().astype(np.uint8).tofile(bin_path) 
        batch = [item.numpy() for item in batch]
        info_path = info_dir/f'image-{idx:0>3}.pkl'
        STAT_FLAGS = os.O_RDWR|os.O_CREAT
        STAT_MODES = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(info_path, STAT_FLAGS, STAT_MODES), 'wb') as fd:
            pickle.dump(batch, fd)


def main():
    config, device, logger, _ = program.preprocess()
    config['Eval']['dataset']['data_dir'] = config['data_dir']
    label_file = Path(config['data_dir'])/'test_icdar2015_label.txt'
    config['Eval']['dataset']['label_file_list'] = [str(label_file)]
    preprocess(config, device, logger)


if __name__ == '__main__':
    main()