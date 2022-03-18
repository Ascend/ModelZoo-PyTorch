# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
import argparse
import os
from solver import Solver
from data_loader import get_loader, get_dist_loader
import random
import torch
import numpy as np

def main(config):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29688'

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    os.environ['PYTHONHASHSEED'] = str(config.seed)

    config.is_master_node = not config.distributed or config.npu_idx == 0

    if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s'%config.model_type)
        return
    
    config.train_path = os.path.join(config.data_path, "train")
    config.valid_path = os.path.join(config.data_path, "valid")
    config.test_path = os.path.join(config.data_path, "test")
    
    # Create directories if not exist
    if config.is_master_node:
        if not os.path.exists(config.model_path):
            os.makedirs(config.model_path)
        if not os.path.exists(config.result_path):
            os.makedirs(config.result_path)
        config.result_path = os.path.join(config.result_path,config.model_type)
        if not os.path.exists(config.result_path):
            os.makedirs(config.result_path)

        print(config)
    
    if config.distributed:
        if 'RANK_SIZE' in os.environ and 'RANK_ID' in os.environ:
            config.rank_size = int(os.environ['RANK_SIZE'])
            config.rank_id = int(os.environ['RANK_ID'])
            config.rank = config.dist_rank * config.rank_size + config.rank_id
            config.world_size = config.world_size * config.rank_size
            config.npu_idx = config.rank_id
            config.batch_size = int(config.batch_size / config.rank_size)
            config.num_workers = int((config.num_workers + config.rank_size - 1) / config.rank_size)
        else:
            raise RuntimeError("init_distributed_mode failed.")

        torch.distributed.init_process_group(backend='hccl', init_method=config.dist_url,
                                             world_size=config.world_size, rank=config.rank)

        train_loader, train_sampler = get_dist_loader(image_path=config.train_path,
                                  image_size=config.image_size,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  mode='train',
                                  augmentation_prob=config.augmentation_prob)

        config.train_sampler = train_sampler
    else:
        config.rank_size = 1
        train_loader = get_loader(image_path=config.train_path,
                                  image_size=config.image_size,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  mode='train',
                                  augmentation_prob=config.augmentation_prob)

    valid_loader = get_loader(image_path=config.valid_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='valid',
                            augmentation_prob=0.)
    test_loader = get_loader(image_path=config.test_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='test',
                            augmentation_prob=0.)

    solver = Solver(config, train_loader, valid_loader, test_loader)


    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
    parser.add_argument('--pth_path', type=str, default='/home/R2U_Net.pkl')

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--data_path', type=str, default='/home/dataset/')
    parser.add_argument('--result_path', type=str, default='./result/')

    parser.add_argument('--npu_idx', type=int, default=1)
    parser.add_argument('--apex', type=int, default=0)
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )
    parser.add_argument('--loss_scale_value',
                        default=1024,
                        type=int,
                        help='set loss scale value.')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed',
                        action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs.')
    parser.add_argument('--dist_rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')
    parser.add_argument('--seed',
                        default=12345,
                        type=int,
                        help='Manually set random seed')
    config = parser.parse_args()
    main(config)
