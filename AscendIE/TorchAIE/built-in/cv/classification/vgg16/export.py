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

import sys
import os
import argparse

import torch
import torchvision.models as models


def parse_args():
    parser = argparse.ArgumentParser(description='Export VGG16 .ts model file')
    parser.add_argument('--model_path',help='VGG16 pth file path', type=str,
                        default='./vgg16-397923af.pth'
                        )
    parser.add_argument('--ts_save_path', help='VGG16 torch script model save path', type=str,
                        default='vgg16.ts')
    
    args = parser.parse_args()
    return args

def check_args(args):
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'VGG16 model file {args.model_path} not exists')
    
def trace_ts_model(model_path, ts_save_path):
    # load model
    model = models.vgg16(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # trace model
    input_data = torch.ones(1, 3, 224, 224)
    ts_model = torch.jit.trace(model, input_data)
    ts_model.save(ts_save_path)
    print(f'VGG16 torch script model saved to {ts_save_path}')


if __name__ == '__main__':
    print('Start to export VGG16 torch script model')
    opts = parse_args()
    check_args(opts)

    # load & trace model
    trace_ts_model(opts.model_path, opts.ts_save_path)
    print("Finish Tracing VGG16 model")