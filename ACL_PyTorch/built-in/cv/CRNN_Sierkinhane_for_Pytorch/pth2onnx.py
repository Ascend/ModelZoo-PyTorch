# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
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

# coding=utf-8

import argparse

import torch

import config
from default_arguments import CHECKPOINT_FILE, CONFIG_FILE
import lib.models.crnn as crnn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=CONFIG_FILE)
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_FILE)
    return parser.parse_args()


def convert(config_filepath, checkpoint_filepath):
    device = torch.device('cpu')
    model = crnn.get_crnn(config.get_config(config_filepath)).to(device)
    __load_checkpoint(model, checkpoint_filepath, device)
    model.eval()
    torch.onnx.export(
        model,
        torch.randn(1, 1, 32, 160),
        'crnn.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: '-1'}, 'output': {1: '-1'}}
    )


def __load_checkpoint(model, checkpoint_filepath, device):
    checkpoint = torch.load(checkpoint_filepath, map_location=device)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)


if __name__ == '__main__':
    args = parse_args()
    convert(args.config, args.checkpoint)
