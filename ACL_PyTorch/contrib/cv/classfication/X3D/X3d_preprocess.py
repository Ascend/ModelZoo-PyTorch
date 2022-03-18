# Copyright 2021 Huawei Technologies Co., Ltd
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

import torch
import numpy as np
import sys
sys.path.append(r"slowfast")
from slowfast.datasets import loader
from slowfast.utils.parser import load_config, parse_args
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job

def perform_x3d_preprocess(test_loader, data_output_path, cfg):
    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        if (cfg.TEST.BATCH_SIZE != 1):
            input = inputs[0]
            if (inputs[0].shape[0] != cfg.TEST.BATCH_SIZE):
                a = cfg.TEST.BATCH_SIZE - inputs[0].shape[0]
                input_add = torch.zeros(a,3,13,182,182)
                input = np.append(input, input_add)
                input = input.reshape(cfg.TEST.BATCH_SIZE,3,13,182,182)
            label_str = ''
            input = torch.tensor(input)
            input = input.numpy()

            for i in range(labels.shape[0]):
                label = labels[i].numpy()
                label_str = label_str + str(label) + ','
            if (labels.shape[0] != cfg.TEST.BATCH_SIZE):
                a = cfg.TEST.BATCH_SIZE - labels.shape[0]
                for i in range(a):
                    label_add = '-1,'
                    label_str = label_str + label_add
            input.tofile(data_output_path + str(cur_iter) + "_" + label_str + ".bin")
            print("construct_input_bin-->index=", cur_iter, "label=", label_str)
        else:
            input = inputs[0]
            label_str = ''
            input = input.numpy()
            for i in range(labels.shape[0]):
                label = labels[i].numpy()
                label_str = label_str + str(label) + ','
            input.tofile(data_output_path + str(cur_iter) + "_" + label_str + ".bin")
            print("construct_input_bin-->index=", cur_iter, "label=", label_str)

def x3d_preprocess(cfg):
    data_output_path = cfg.X3D_PREPROCESS.DATA_OUTPUT_PATH
    test_loader = loader.construct_loader(cfg, "test")
    perform_x3d_preprocess(test_loader, data_output_path, cfg)

if __name__== '__main__':
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    launch_job(cfg=cfg, init_method=args.init_method, func=x3d_preprocess)

