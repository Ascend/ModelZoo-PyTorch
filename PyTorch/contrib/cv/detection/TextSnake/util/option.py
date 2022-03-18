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
import torch
import os
import torch.backends.cudnn as cudnn

from datetime import datetime

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def arg2str(args):
    args_dict = vars(args)
    option_str = datetime.now().strftime('%b%d_%H-%M-%S') + '\n'

    for k, v in sorted(args_dict.items()):
        option_str += ('{}: {}\n'.format(str(k), str(v)))

    return option_str

class BaseOptions(object):

    def __init__(self):

        self.parser = argparse.ArgumentParser()

        # basic opts
        self.parser.add_argument('exp_name', type=str, help='Experiment name')
        self.parser.add_argument('--net', default='vgg', type=str, choices=['vgg', 'resnet'], help='Network architecture')
        self.parser.add_argument('--dataset', default='total-text', type=str, choices=['synth-text', 'total-text'], help='Dataset name')
        self.parser.add_argument('--resume', default=None, type=str, help='Path to target resume checkpoint')
        self.parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
        self.parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
        
        self.parser.add_argument('--mgpu', action='store_true', help='Use multi-gpu to train model')
        self.parser.add_argument('--save_dir', default='./save/', help='Path to save checkpoint models')
        self.parser.add_argument('--vis_dir', default='./vis/', help='Path to save visualization images')
        self.parser.add_argument('--log_dir', default='./logs/', help='Path to tensorboard log')
        self.parser.add_argument('--loss', default='CrossEntropyLoss', type=str, help='Training Loss')
        self.parser.add_argument('--input_channel', default=1, type=int, help='number of input channels' )
        self.parser.add_argument('--pretrain', default=False, type=str2bool, help='Pretrained AutoEncoder model')
        self.parser.add_argument('--verbose', '-v', default=True, type=str2bool, help='Whether to output debug info')
        self.parser.add_argument('--viz', action='store_true', help='Whether to output debug info')

        # train opts
        self.parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
        self.parser.add_argument('--max_epoch', default=500, type=int, help='Max epochs')
        self.parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
        self.parser.add_argument('--lr_adjust', default='fix', choices=['fix', 'poly'], type=str, help='Learning Rate Adjust Strategy')
        self.parser.add_argument('--stepvalues', default=[], nargs='+', type=int, help='# of iter to change lr')
        self.parser.add_argument('--weight_decay', '--wd', default=0., type=float, help='Weight decay for SGD')
        self.parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD lr')
        self.parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
        self.parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
        self.parser.add_argument('--optim', default='SGD', type=str, choices=['SGD', 'Adam'], help='Optimizer')
        self.parser.add_argument('--display_freq', default=50, type=int, help='display training metrics every # iterations')
        self.parser.add_argument('--viz_freq', default=50, type=int, help='visualize training process every # iterations')
        self.parser.add_argument('--save_freq', default=10, type=int, help='save weights every # epoch')
        self.parser.add_argument('--log_freq', default=100, type=int, help='log to tensorboard every # iterations')
        self.parser.add_argument('--val_freq', default=100, type=int, help='do validation every # iterations')

        # data args
        self.parser.add_argument('--rescale', type=float, default=255.0, help='rescale factor')
        self.parser.add_argument('--means', type=int, default=(0.485, 0.456, 0.406), nargs='+', help='mean')
        self.parser.add_argument('--stds', type=int, default=(0.229, 0.224, 0.225), nargs='+', help='std')
        self.parser.add_argument('--input_size', default=512, type=int, help='model input size')

        # eval args
        self.parser.add_argument('--checkepoch', default=-1, type=int, help='Load checkpoint number')

        # demo args
        self.parser.add_argument('--img_root', default=None, type=str, help='Path to deploy images')

        self.parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
        self.parser.add_argument('-g', '--gpus', default=1, type=int,
                            help='number of gpus per node')
        self.parser.add_argument('-nr', '--rank', default=0, type=int,
                            help='ranking within the nodes')

        # npu args
        self.parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
        self.parser.add_argument('--device_list', default='0,1,2,3,4,5,6,7', type=str, help='device id list')
        self.parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
        self.parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
        self.parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
        self.parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

    def parse(self, fixed=None):

        if fixed is not None:
            args = self.parser.parse_args(fixed)
        else:
            args = self.parser.parse_args()

        return args

    def initialize(self, fixed=None):

        # Parse options
        self.args = self.parse(fixed)

        # Setting default torch Tensor type
        #if self.args.cuda and torch.cuda.is_available():
        #    torch.set_default_tensor_type('torch.cuda.FloatTensor')
        #    cudnn.benchmark = True
        #else:
        #    torch.set_default_tensor_type('torch.FloatTensor')

        # Create weights saving directory
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        # Create weights saving directory of target model
        model_save_path = os.path.join(self.args.save_dir, self.args.exp_name)

        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

        return self.args

    def update(self, args, extra_options):

        for k, v in extra_options.items():
            setattr(args, k, v)
