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
# limitations under the License.

# -*- coding: utf-8 -*-
"""用于精度比对
"""

import torch
import torch.nn as nn
import torchvision
from apex import amp
import copy


##### 需自行改写部分 start #####
import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv.runner import init_dist,IterBasedRunner, build_optimizer, Fp16OptimizerHook
from mmcv.utils import Config, DictAction, get_git_hash
 
from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu_ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--apex_opt_level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )
    # loss_scale_value == -1 表示None
    parser.add_argument('--loss_scale_value',
                        default=128,
                        type=int,
                        help='set loss scale value.')

    parser.add_argument(
        '--use_amp',
        type=str2bool,
        nargs='?',
        const=True,
        default=None,
        help='use nvidia apex amp ?'
    )
    parser.add_argument(
        '--sys_fp_16',
        type=str2bool,
        nargs='?',
        const=True,
        default=None,
        help='use sys fp16 ?'
    )
    
    parser.add_argument(
        '--use_npu',
        type=str2bool,
        nargs='?',
        const=True,
        default=None,
        help='use 华为 npu ?'
    )
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


args = parse_args()
if args.resume_from is not None:
    args.work_dir = '/'.join(args.resume_from.split('/')[:-1])
cfg = Config.fromfile(args.config)
# cfg.model.pretrained = True
print(args.use_npu)
print(type(args.use_npu))

if args.options is not None:
    cfg.merge_from_dict(args.options)
# set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True
# npu 服务器设置
if args.use_npu is not None:
    cfg.use_npu = args.use_npu
elif cfg.get('use_npu',None) is None:
    cfg.use_npu = False
if cfg.use_npu:
    cfg.dist_params.backend = 'hccl'
else:
    cfg.dist_params.backend = 'nccl'
    os.environ["NCCL_DEBUG"] = "INFO"

# 添加从命令行控制amp的方法
if args.use_amp is not None:
    cfg.use_amp = args.use_amp
elif cfg.get('use_amp',None) is None:
    cfg.use_amp = False

if args.sys_fp_16 is not None:
    cfg.sys_fp_16 = args.sys_fp_16
elif cfg.get('sys_fp_16',None) is None:
    cfg.sys_fp_16 = False


print("work_dir", args.work_dir)
cfg.apex_opt_level = args.apex_opt_level
cfg.loss_scale_value = args.loss_scale_value
if cfg.loss_scale_value == -1:
    cfg.loss_scale_value = None

# work_dir is determined in this priority: CLI > segment in file > filename
if args.work_dir is not None:
    # update configs according to CLI args if args.work_dir is not None
    cfg.work_dir = args.work_dir
elif cfg.get('work_dir', None) is None:
    # use config filename as default work_dir if cfg.work_dir is None
    cfg.work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
if args.load_from is not None:
    cfg.load_from = args.load_from
if args.resume_from is not None:
    cfg.resume_from = args.resume_from
if args.gpu_ids is not None:
    cfg.gpu_ids = args.gpu_ids
    CALCULATE_DEVICE='npu:'+str(cfg.gpu_ids[0])
    torch.npu.set_device(CALCULATE_DEVICE)
else:
    cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

print(cfg.gpu_ids)
print(len(cfg.gpu_ids))

# init distributed env first, since logger depends on the dist info.

distributed = False
# create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
# dump config
cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
# init the logger before other steps
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

# init the meta dict to record some important information such as
# environment info and seed, which will be logged
meta = dict()
# log env info
env_info_dict = collect_env()
env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
dash_line = '-' * 60 + '\n'
logger.info('Environment info:\n' + dash_line + env_info + '\n' +
            dash_line)
meta['env_info'] = env_info

# log some basic info
logger.info(f'Distributed training: {distributed}')
logger.info(f'Config:\n{cfg.pretty_text}')

# set random seeds
if args.seed is not None:
    logger.info(f'Set random seed to {args.seed}, deterministic: '
                f'{args.deterministic}')
    set_random_seed(args.seed, args.use_npu,deterministic=args.deterministic)
cfg.seed = args.seed
meta['seed'] = args.seed
meta['exp_name'] = osp.basename(args.config)


dataset = build_dataset(cfg.data.train)
dataloader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            seed=cfg.seed,
            drop_last=True,
            pin_memory=False)

# 获得模型
def get_model():

    model = build_segmentor(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    # 用于避免BN或者Dropout带来的影响
    model.train()
    for name,parameters in model.named_parameters():
        print(name,':',parameters)
    
    return model

# 设置npu_device
npu_device = 'npu:' + str(cfg.gpu_ids[0])
test_iter = 1
# 设置amp
AMP_MODE = True

# 设置NPU prof 文件输出
NPU_PROF = False

##### 需自行改写部分 end #####

def cri_func(x):
    base_func = nn.CrossEntropyLoss()
    shape_list = x.shape
    N = shape_list[0]
    R = 1
    if len(shape_list) > 1:
        for r in shape_list[1:]:
            R *= r
    T = torch.randint(0,R, size=(N,)).to(x.device)
    if str(T.device).startswith('npu'):
        T = T.int()
    return base_func(x.reshape(N, -1), T)

# 设置hook
def hook_func(name, save_dict, module):
    def hook_function(module, inputs, outputs):
        inputs_key = name + '_inputs'
        idx = 0
        while inputs_key in save_dict:
            inputs_key = inputs_key.split('-')[0] + '-%d'%idx
            idx +=1
        save_dict[inputs_key] = inputs
        # torch.save(inputs, f"./all_input/{inputs_key}.pt")
        outputs_key = name + '_outputs'
        idx = 0
        while outputs_key in save_dict:
            outputs_key = outputs_key.split('-')[0] + '-%d'%idx
            idx +=1
        save_dict[outputs_key] = outputs
    return hook_function

from mmcv.parallel import scatter_kwargs
def sk(devices,*inputs, **kwargs):
    return scatter_kwargs(inputs, kwargs, devices)

##### CPU #####
# CPU固定输入和权重
model = get_model()
optimizer = build_optimizer(model, cfg.optimizer)
state_dict = copy.deepcopy(model.state_dict())

# CPU注册hook，cpu_dict用于存储对比对象
cpu_dict = {}
for name, module in model.named_modules():
    module.register_forward_hook(hook_func('[forward]:' + name, cpu_dict, module))
    module.register_backward_hook(hook_func('[backward]:' + name, cpu_dict, module))

# CPU运行正反向，获取正反向每个module的输入输出和所有参数的grad
cpu_device = 'cpu'
model.to(cpu_device)
model = model.float()
# 获得输入tensor
# input_tensor = torch.randn(1, 3, 768, 768)
iter_loader = iter(dataloader)
print("cpu input data")
for _ in range(test_iter):
    input_tensor = next(iter_loader)
    print("cpu - input")
    print(input_tensor)
    jz = input_tensor
    inputs, kwargs = sk([cpu_device], input_tensor, optimizer)
    out = model.train_step(*inputs[0], **kwargs[0])
    print(f"finish on cpu forward {_}")
    loss = out['loss']
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
for name, param in model.named_parameters():
    cpu_dict["[grad]:" + name] = param.grad
print("finish on cpu")

##### NPU #####
# 重新定义模型，清理模型状态，并加装权重，保持初始化一致
model = get_model()
optimizer = build_optimizer(model, cfg.optimizer)
model.load_state_dict(state_dict)

# NPU注册hook，npu_dict用于存储对比对象
npu_dict = {}
for name, module in model.named_modules():
    module.register_forward_hook(hook_func('[forward]:' + name, npu_dict, module))
    module.register_backward_hook(hook_func('[backward]:' + name, npu_dict, module))

# 将model和input_tensor放到npu
torch.npu.set_device(npu_device)
model = model.to(npu_device)

print('npu data input')

# amp可选项，不适用请注释
if AMP_MODE:
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1', loss_scale=1.0)

# NPU运行正反向，获取正反向每个module的输入输出和所有参数的grad
dataset = build_dataset(cfg.data.train)
dataloader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            seed=cfg.seed,
            drop_last=True,
            pin_memory=False)

iter_loader = iter(dataloader)
for _ in range(test_iter):
    input_tensor = next(iter_loader)
    print("npu - input")
    print(jz)
    inputs, kwargs = sk([npu_device], jz, optimizer)
    out = model.train_step(*inputs[0], **kwargs[0])

    print(f"finish on npu forward {_}")
    loss = out['loss']
    optimizer.zero_grad()
    if AMP_MODE:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()
print("finish on npu backward")
for name, param in model.named_parameters():
    npu_dict["[grad]:" + name] = param.grad

##### ComPare #####
# 递归得到对比值
def compare(x1, x2, prefix=''):
    if isinstance(x1, tuple):
        if x1:
            for idx in range(len(x1)):
                try:
                    compare(x1[idx], x2[idx], prefix=prefix + '.%d' % idx)
                except Exception as e:
                    # print(str(e))
                    print(prefix, 'failed.')
    elif isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
        try:
            # print("*" * 20)
            # print(x2.cpu())
            # print("*" * 20)
            l1_error = (x1 - x2.cpu()).abs().mean()
            rel_error = l1_error / (x1.abs().mean())
            if "[backward]:backbone.blocks.0" in prefix or "[backward]:backbone.blocks.1." in prefix :
                print("*" * 10, "npu", "*"*10)
                print(x2.cpu())
                print("min:", x2.cpu().min(), "max:", x2.cpu().max(), "mean:", x2.cpu().mean())
                print("*" * 10)
                print("*" * 10, "cpu", "*"*10)
                print(x1.cpu())
                print("min:", x1.cpu().min(), "max:", x1.cpu().max(), "mean:", x1.cpu().mean())
                print("*" * 10)
            print(prefix, 'l1_error: ', l1_error, 'rel_error', rel_error)
            if l1_error * rel_error > 10 :
                print('\n###\n',prefix, 'should checked!','\n###\n')
        except Exception as e:
            print(x1.dtype, "="*10, x2.dtype)
            print(str(e))
            print(prefix, 'failed.')

for k in cpu_dict:
    compare(cpu_dict[k], npu_dict[k], prefix=k)

# 需要profiling的时候额外输出一次
if NPU_PROF:
    with torch.autograd.profiler.profile(use_npu=True) as prof:
        out = model.train_step(*inputs[0], **kwargs[0])
        loss = out['loss']
        optimizer.zero_grad()
        if AMP_MODE:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
    prof.export_chrome_trace("output.prof")  # "output.prof"为输出文件地址

