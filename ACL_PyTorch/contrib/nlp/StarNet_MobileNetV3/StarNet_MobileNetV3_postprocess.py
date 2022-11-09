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

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, 'PaddleOCR')))

import paddle
import numpy as np
import tools.program as program

from tqdm import tqdm
from ppocr.data import build_dataloader
from ppocr.metrics import build_metric
from ppocr.postprocess import build_post_process


def main(config, device, logger, vdl_writer):
    valid_dataloader = build_dataloader(config, 'Eval', device, logger)

    eval_class = build_metric(config['Metric'])

    global_config = config['Global']
    post_process_class = build_post_process(config['PostProcess'], global_config)
    
    pbar = tqdm(
            total=len(valid_dataloader),
            desc='Postprocessing',
            position=0,
            leave=True)
            
    for idx, batch in enumerate(valid_dataloader):
            result_name = "img_{}_0.bin".format(idx)           
            result_path = os.path.join(__dir__, config['results'], result_name)
            
            preds = paddle.to_tensor(np.fromfile(result_path, dtype=np.float32).reshape(1, 25, 37))
                        
            batch = [item.numpy() for item in batch]        
            post_result = post_process_class(preds, batch[1])
            
            eval_class(post_result, batch)
            
            pbar.update(1)
            
    pbar.close()
    metric = eval_class.get_metric()
    print(metric)
    

if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess()  
    main(config, device, logger, vdl_writer)
