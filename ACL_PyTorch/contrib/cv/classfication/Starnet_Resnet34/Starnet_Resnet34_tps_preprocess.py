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
import numpy as np
from tqdm import tqdm
import tools.program as program
from ppocr.data import build_dataloader

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, 'PaddleOCR')))
def main(configs, devices, loggers, vdl_writers, data_paths):
    valid_dataloader = build_dataloader(config, 'Eval', device, logger)

    pbar = tqdm(
            total=len(valid_dataloader),
            desc='Preprocessing',
            position=0,
            leave=True)
    
    for idx, batch in enumerate(valid_dataloader):
        img_name = 'img_{}.bin'.format(idx)
        
        batch[0].numpy().tofile(os.path.join(data_path, img_name))
        
        pbar.update(1)
    
    pbar.close()         


if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess()
    
    data_path = os.path.join(__dir__, config['bin_data'])

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    main(config, device, logger, vdl_writer, data_path)
