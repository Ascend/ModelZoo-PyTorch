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


import tqdm
import pickle
from pathlib import Path

import numpy as np

from ppocr.data import build_dataloader
import tools.program as program


def preprocess(config, device, logger):

    valid_dataloader = build_dataloader(config, 'Eval', device, logger)

    bin_dir = Path(config['bin_dir'])
    info_dir = Path(config['info_dir'])
    bin_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)

    for idx, batch in enumerate(tqdm.tqdm(valid_dataloader, desc='Processing')):
        image = batch[0]
        bin_path = bin_dir/f'image-{idx:0>6}.bin'
        image.numpy().astype(np.float32).tofile(bin_path)

        batch = [item.numpy() for item in batch]
        info_path = info_dir/f'image-{idx:0>6}.pkl'
        with open(info_path, 'wb') as f:
            pickle.dump(batch, f)


def main():

    config, device, logger, _ = program.preprocess()
    config['Eval']['dataset']['data_dir'] = config['data_dir']
    preprocess(config, device, logger)


if __name__ == '__main__':
    main()
