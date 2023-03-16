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
import pickle
from pathlib import Path

import tqdm
import paddle
import numpy as np

sys.path.append(os.path.abspath('./PaddleOCR'))
from ppocr.data import create_operators, transform
from ppocr.utils.utility import get_image_file_list
import tools.program as program

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'


def main():

    global_config = config['Global']

    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image', 'shape']
        transforms.append(op)

    assert 'NormalizeImage' in transforms[2]
    # remove NormalizeImage because AIPP can do same thing
    transforms.pop(2)
    assert 'ToCHWImage' in transforms[2]
    # remove ToCHWImage because AIPP require HWC
    transforms.pop(2)

    ops = create_operators(transforms, global_config)

    pre_dir = Path(config['prep_dir'])
    bin_dir = pre_dir/'img_npy'
    bin_dir.mkdir(parents=True, exist_ok=True)
    info_path = pre_dir/'img_info.pkl'

    img_list = get_image_file_list(config['Global']['infer_img'])
    info_dict = {}

    for file in tqdm.tqdm(img_list):
        with open(file, 'rb') as f:
            img = f.read()
            data = {'image': img}
        batch = transform(data, ops)
        images = np.expand_dims(batch[0], axis=0).astype(np.uint8)
        shape_list = np.expand_dims(batch[1], axis=0)
        bin_path = bin_dir/f'{Path(file).stem}.npy'
        np.save(bin_path, images)
        info_dict[Path(file).name] = shape_list

    with open(info_path, 'wb') as f:
        pickle.dump(info_dict, f)


if __name__ == '__main__':
    config, *_ = program.preprocess()
    main()
