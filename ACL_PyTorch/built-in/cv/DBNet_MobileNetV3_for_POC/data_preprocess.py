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


from pathlib import Path
import platform

import numpy as np
from tqdm import tqdm

from ppocr.data import build_dataloader
import tools.program as program


def data_preprocess(config, logger, with_aipp=True):
    bin_dir = Path(config['bin_dir'])
    info_dir = Path(config['info_dir'])
    bin_dir.mkdir(parents=True, exist_ok=True, mode=644)
    info_dir.mkdir(parents=True, exist_ok=True, mode=644)

    if with_aipp:
        transforms_config = config['Eval']['dataset']['transforms' ]
        for t in transforms_config:
            if 'NormalizeImage' in t:
                # remove NormalizeImage because AIPP can do same thing
                t.pop('NormalizeImage')
            if 'ToCHWImage' in t:
                # remove ToCHWImage because AIPP require HWC
                t.pop('ToCHWImage')
    config['Eval']['dataset']['transforms'] = [t for t in transforms_config if t]

    valid_dataloader = build_dataloader(config, 'Eval', None, logger)
    max_iter = len(valid_dataloader) -1 if platform.system() == "Windows" \
                    else len(valid_dataloader)

    total_size = len(valid_dataloader)
    logger.info(f'Total: {total_size} images.')
    pbar = tqdm(total=total_size, desc='eval model', leave=True)
    for i, batch in enumerate(valid_dataloader):
        if i >= max_iter:
            break
        batch_numpy = [item.numpy() for item in batch]
        if with_aipp:
            batch_numpy[0] = batch_numpy[0].astype(np.uint8)
        batch_numpy[0].tofile(str(bin_dir / f'image-{i:0>3}.bin'))
        np.savez(str(info_dir / f'info-{i:0>3}.npz'),
                 {f'info{j}': value for j, value in enumerate(batch_numpy)})
        pbar.update(1)


def main():
    config, _, logger, _ = program.preprocess()
    config['Eval']['dataset']['data_dir'] = config['data_dir']
    label_file = Path(config['data_dir'])/'test_icdar2015_label.txt'
    config['Eval']['dataset']['label_file_list'] = [str(label_file)]
    data_preprocess(config, logger)


if __name__ == '__main__':
    main()
