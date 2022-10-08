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
from ppocr.postprocess import build_post_process
from ppocr.utils.utility import get_image_file_list


def main(config, device, logger, vdl_writer):
    global_config = config['Global']
    
    post_process_class = build_post_process(config['PostProcess'], global_config)
    
    pbar = tqdm(
            total=5,
            desc='Postprocessing',
            position=0,
            leave=True)
            
    post_results = {}
    for idx in range(1, 6):
        result = os.path.join(config['Global']['infer_results'], 'word_{}_0.bin'.format(idx))
          
        pred = paddle.to_tensor(np.fromfile(result, dtype=np.float32).reshape(1, 40, 97))
        post_result = post_process_class(pred)      
        
        img_name = 'word_{}.png'.format(idx)
        post_results[img_name] = post_result[0]
        
        pbar.update(1)
    
    pbar.close()

    print("Infer Results: ", post_results)


if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess()  
    main(config, device, logger, vdl_writer)
