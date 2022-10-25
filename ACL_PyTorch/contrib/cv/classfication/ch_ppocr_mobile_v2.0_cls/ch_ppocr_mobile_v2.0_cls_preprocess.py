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

import cv2
import shutil
import paddle
import numpy as np
import tools.program as program

from tqdm import tqdm
from ppocr.data import create_operators, transform
from ppocr.utils.utility import get_image_file_list


def main(config, device, logger, vdl_writer, data_path):
    global_config = config['Global']      
    global_config['infer_mode'] = True

    pbar = tqdm(
            total=len(get_image_file_list(config['Global']['infer_img'])),
            desc='Postprocessing',
            position=0,
            leave=True) 
                
    for im_file in get_image_file_list(config['Global']['infer_img']):
        with open(im_file, 'rb') as f:
            img = f.read()
            
        img = np.frombuffer(img, dtype='uint8')
        img = cv2.imdecode(img, 1)
        
        resized_image = cv2.resize(img, (192, 48))

        im_name = '{}.bin'.format(os.path.basename(im_file)[:-4])
        resized_image.tofile(os.path.join(data_path, im_name))
 
        pbar.update(1)
    
    pbar.close()
    print("Preproced Data Saved To: {}".format(data_path))


if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess()  
    
    data_path = os.path.join(__dir__, 'pre_data')
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)
    
    main(config, device, logger, vdl_writer, data_path)
