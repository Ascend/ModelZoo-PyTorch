# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Claiuse License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random

import torch
import numpy as np
from PIL import Image

from data.base_dataset import get_params, get_transform
from options.test_options import TestOptions


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_torch()


def preprocess(opt_class, AB):
    w, h = AB.size
    w2 = int(w / 2)
    A = AB.crop((0, 0, w2, h))
    B = AB.crop((w2, 0, w, h))
    transform_params = get_params(opt_class, A.size)
    B_transform = get_transform(opt_class, transform_params, grayscale=(3 == 1))
    B = B_transform(B)
    return  B# 0.9686 # A[1,35,46] tensor(-0.1451)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    src_path = os.path.join(opt.dataroot, 'test')
    save_path = opt.results_dir
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    in_files = os.listdir(src_path)
  
  
    for idx, filename in enumerate(in_files):
        idx = idx + 1
        print(filename, "===", idx)
        input_image = Image.open(src_path + '/' + filename).convert('RGB')
        input_tensor = preprocess(opt, input_image)
        img = np.array(input_tensor).astype(np.float32)
        img.tofile(os.path.join(save_path, filename.split('.')[0] + ".bin"))
