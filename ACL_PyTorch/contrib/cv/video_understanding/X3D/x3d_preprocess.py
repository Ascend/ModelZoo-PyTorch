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
import os.path as osp

from tqdm import tqdm
import numpy as np
import torch

sys.path.append(r"slowfast")
from slowfast.datasets import loader
from slowfast.utils.parser import load_config, parse_args
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job


def x3d_preprocess(cfg):
    cfg.TEST.BATCH_SIZE = 1
    data_output_path = cfg.X3D_PREPROCESS.DATA_OUTPUT_PATH
    test_loader = loader.construct_loader(cfg, "test")
    if not osp.isdir(data_output_path):
        os.mkdir(data_output_path)
    for cur_iter, (inputs, labels, *_) in tqdm(enumerate(test_loader)):
        input = inputs[0].numpy()
        input.tofile(osp.join(data_output_path, f'{cur_iter}_{labels[0]}.bin'))


if __name__== '__main__':
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    launch_job(cfg=cfg, init_method=args.init_method, func=x3d_preprocess)
