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

import os
import argparse
import torch.multiprocessing as mp
import random
import torch
if torch.__version__ >= "1.8":
    import torch_npu
from solver import Solver

from torch.backends import cudnn

import warnings

warnings.filterwarnings("ignore")
#CALCULATE_DEVICE = "npu:0"

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Solver for training and testing StarGAN.
    solver = Solver(config)

    if config.distributed:
        # if config.mode == 'train':
        mp.spawn(solver.train, nprocs = config.npus, args = (config.npus,))
        # elif config.mode == 'test':
        #     mp.spawn(solver.test,  nprocs = config.gpus, args = (config.gpus,), join = True)
    else :
        solver.train()



if __name__ == '__main__':

#    if 'npu' in CALCULATE_DEVICE:
#       torch.npu.set_device(CALCULATE_DEVICE)

    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    
    # Training configuration.
    parser.add_argument('--epoch', type=int, default=20, help='Training Epoch')
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])


    # DDP.
    parser.add_argument('--npus', type = int, default = 1)
    parser.add_argument('--distributed', type = bool, default = False)

    # Amp
    parser.add_argument('--amp', default = False, action="store_true")

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    
    # Directories.
    parser.add_argument('--dataset_dir', type=str, default='data')
    parser.add_argument('--folder_dir', type=str, default='/home/cly/StarGAN/stargan')

    # Step size.
    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(random.randrange(1001, 49999))

    config = parser.parse_args()

    config.celeba_image_dir = config.dataset_dir + "/celeba/images"
    config.attr_path = config.dataset_dir + "/celeba/list_attr_celeba.txt"
    config.log_dir = config.folder_dir + "/log"
    config.model_save_dir = config.folder_dir + "/models"
    config.sample_dir = config.folder_dir + "/samples"
    config.result_dir = config.folder_dir + "/results"
    config.dataset = "CelebA"

    print(config)
    main(config)
