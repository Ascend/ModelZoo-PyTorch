# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import shutil
import pprint

import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import _init_paths
import models
from config import config
from config import update_config
from core.function import validate
from utils.modelsummary import get_model_summary
from utils.utils import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('data_path',
                        help='data path',
                        type=str)

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')

    parser.add_argument('--bs',
                        help='batch_size',
                        type=int,
                        default='')

    parser.add_argument('--nproc',
                        help='nproc',
                        type=int,
                        default='')
    
    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    if os.getenv('ALLOW_FP32', False) and os.getenv('ALLOW_HF32', False):
        raise RuntimeError('ALLOW_FP32 and ALLOW_HF32 cannot be set at the same time!')
    elif os.getenv('ALLOW_HF32', False):
        torch.npu.conv.allow_hf32 = True
    elif os.getenv('ALLOW_FP32', False):
        torch.npu.conv.allow_hf32 = False
        torch.npu.matmul.allow_hf32 = False
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(
        config)

    dump_input = torch.rand(
        (1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
    )
    #logger.info(get_model_summary(model, dump_input))

    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = model.npu()
    bs = args.bs
    nproc = args.nproc
    
    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().npu()

    # Data loading code
    data_path = args.data_path
    valdir = os.path.join(data_path, "val")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    valid_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=bs,
        shuffle=False,
        num_workers=nproc,
        pin_memory=True
    )

    # evaluate on validation set
    validate(config, valid_loader, model, criterion, final_output_dir,
             tb_log_dir, None)


if __name__ == '__main__':
    main()
