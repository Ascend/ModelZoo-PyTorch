# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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

import argparse
import os
import sys

import torch

sys.path.append(r"./HRNet-Image-Classification")
sys.path.append(r"./HRNet-Image-Classification/lib")
from lib.models import cls_hrnet
from lib.config import config
from lib.config import update_config


def parse_args():
    parser = argparse.ArgumentParser(description='Export HRNet .ts model file')
    parser.add_argument('--cfg', help='experiment configure file name', type=str,
                        default='./HRNet-Image-Classification/experiments/'
                                'cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
                        )
    parser.add_argument('--model_path', help='HRNet  pth file path', type=str,
                        default='./hrnetv2_w18_imagenet_pretrained.pth'
                        )
    parser.add_argument('--ts_save_path', help='HRNet torch script model save path', type=str,
                        default='hrnet.pt')

    # dummy parameters for HRNet config setting
    parser.add_argument('--modelDir', type=bool, default=False, help='dummy parameter for HRNet config setting')
    parser.add_argument('--dataDir', type=bool, default=False, help='dummy parameter for HRNet config setting')
    parser.add_argument('--testModel', type=bool, default=False, help='dummy parameter for HRNet config setting')
    parser.add_argument('--logDir', type=bool, default=False, help='dummy parameter for HRNet config setting')
    args = parser.parse_args()
    return args


def check_args(args):
    if not os.path.exists(args.cfg):
        raise FileNotFoundError(f'config file {args.cfg} not exists')
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'HRNet model file {args.model_path} not exists')


if __name__ == '__main__':
    print('Start to export HRNet torch script model')
    opts = parse_args()
    check_args(opts)

    # load HRNet model
    checkpoint = torch.load(opts.model_path, map_location='cpu')
    update_config(config, opts)
    model = cls_hrnet.get_cls_net(config)
    model.load_state_dict(checkpoint)
    model.eval()

    # save torch script model
    input_data = torch.ones(1, 3, 224, 224)
    ts_model = torch.jit.script(model, input_data)
    ts_model.save(opts.ts_save_path)
    print(f'HRNet torch script model saved to {opts.ts_save_path}')
