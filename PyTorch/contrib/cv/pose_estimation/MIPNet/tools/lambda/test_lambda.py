# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsLambdaMSELoss
from core.loss import JointsMSELoss
from core.validate import validate_lambda_quantitative
from core.validate import validate_lambda 
from utils.utils import create_logger
from utils.utils import get_lambda_model_summary
from utils.utils import set_seed

import dataset
import models

# --------------------------------------------------------------------------------
set_seed(seed_id=0)

# --------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

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
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    dump_lambda = torch.rand(
        (1, 2)
    )

    logger.info(get_lambda_model_summary(model, dump_input, dump_lambda))

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model_object = torch.load(cfg.TEST.MODEL_FILE)
        if 'latest_state_dict' in model_object.keys():
            logger.info('=> loading from latest_state_dict at {}'.format(cfg.TEST.MODEL_FILE))
            model.load_state_dict(model_object['latest_state_dict'], strict=False)
        else:
            logger.info('=> no latest_state_dict found')
            model.load_state_dict(model_object, strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model = torch.nn.DataParallel(model).cuda()
    
    # ------------------------------------------------

    # define loss function (criterion) and optimizer
    criterion_lambda = JointsLambdaMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.TEST_DATASET)(
        cfg=cfg, image_dir=cfg.DATASET.TEST_IMAGE_DIR, annotation_file=cfg.DATASET.TEST_ANNOTATION_FILE, \
        dataset_type=cfg.DATASET.TEST_DATASET_TYPE, \
        image_set=cfg.DATASET.TEST_SET, is_train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # # # evaluate on validation set
    validate_lambda_quantitative(cfg, valid_loader, valid_dataset, model, criterion,
             os.path.join(final_output_dir, 'lambda:0,1'), tb_log_dir, writer_dict, print_prefix='lambda', lambda_vals=[0, 1])

    # # # evaluate on validation set
    # validate_lambda(cfg, valid_loader, valid_dataset, model, criterion_lambda, criterion,
    #          final_output_dir, tb_log_dir, writer_dict, print_prefix='lambda')


    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
