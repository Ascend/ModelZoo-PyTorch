#
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
#
import os
import tqdm
from os.path import dirname

import torch.backends.cudnn as cudnn
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')
cudnn.benchmark = True
cudnn.enabled = True

import torch
import importlib
import argparse
from datetime import datetime
from pytz import timezone
import time
import shutil

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--continue_exp', type=str, help='continue exp')
    parser.add_argument('-e', '--exp', type=str, default='pose', help='experiments name')
    parser.add_argument('-m', '--max_iters', type=int, default=250, help='max number of iterations (thousands)')
    parser.add_argument( '--ddp',default=False,help='distributed or not ')
    args = parser.parse_args()
    if args.ddp:
        NPU_WORLD_SIZE = int(os.getenv('NPU_WORLD_SIZE'))
        RANK = int(os.getenv('RANK'))
        torch.distributed.init_process_group('hccl', rank=RANK, world_size=NPU_WORLD_SIZE)
    return args

def reload(config):
    """
    load or initialize model's parameters by config from config['opt'].continue_exp
    config['train']['epoch'] records the epoch num
    config['inference']['net'] is the model
    """
    opt = config['opt']

    if opt.continue_exp:
        resume = os.path.join('exp', opt.continue_exp)
        resume_file = os.path.join(resume, 'checkpoint.pt')
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume_file)

            config['inference']['net'].load_state_dict(checkpoint['state_dict'])
            config['train']['optimizer'].load_state_dict(checkpoint['optimizer'])
            config['train']['epoch'] = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            exit(0)

    if 'epoch' not in config['train']:
        config['train']['epoch'] = 0

def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    """
    from pytorch/examples
    """
    basename = dirname(filename)
    if not os.path.exists(basename):
        os.makedirs(basename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pt')

def save(config):
    resume = os.path.join('exp', config['opt'].exp)
    if config['opt'].exp=='pose' and config['opt'].continue_exp is not None:
        resume = os.path.join('exp', config['opt'].continue_exp)
    resume_file = os.path.join(resume, 'checkpoint.pt')

    save_checkpoint({
            'state_dict': config['inference']['net'].state_dict(),
            'optimizer' : config['train']['optimizer'].state_dict(),
            'epoch': config['train']['epoch'],
        }, False, filename=resume_file)
    print('=> save checkpoint')

def train(train_func, data_func, config, post_epoch=None):
    while True:
        fails = 0
        print('epoch: ', config['train']['epoch'])
        if 'epoch_num' in config['train']:
            if config['train']['epoch'] > config['train']['epoch_num']:
                break

        for phase in ['train', 'valid']:
            num_step = config['train']['{}_iters'.format(phase)]
            generator = data_func(phase)
            print('start', phase, config['opt'].exp)

            show_range = range(num_step)
            show_range = tqdm.tqdm(show_range, total = num_step, ascii=True)
            batch_id = num_step * config['train']['epoch']
            if batch_id > config['opt'].max_iters * 1000:
                return
            for i in show_range:
                datas = next(generator)
                outs = train_func(batch_id + i, config, phase, **datas)
        config['train']['epoch'] += 1
        save(config)

def init():
    """
    task.__config__ contains the variables that control the training and testing
    make_network builds a function which can do forward and backward propagation
    """
    opt = parse_command_line()
    task = importlib.import_module('task.pose')
    exp_path = os.path.join('exp', opt.exp)
    
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    config = task.__config__
    try: os.makedirs(exp_path)
    except FileExistsError: pass

    config['opt'] = opt
    config['data_provider'] = importlib.import_module(config['data_provider'])

    func = task.make_network(config)
    reload(config)
    return func, config

def main():
    func, config = init()
    data_func = config['data_provider'].init(config)
    train(func, data_func, config)
    print(datetime.now(timezone('EST')))

if __name__ == '__main__':
    main()
