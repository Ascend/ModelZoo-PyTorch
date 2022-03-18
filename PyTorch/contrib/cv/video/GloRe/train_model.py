# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
import logging

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from data import iterator_factory
from train import metric
from train import custom_optim
from train.model import model
from train.lr_scheduler import MultiFactorScheduler


def train_model(sym_net, model_prefix, dataset, input_conf,
                clip_length=8, train_frame_interval=2, val_frame_interval=2,
                resume_epoch=-1, batch_size=4, save_frequency=1,
                lr_base=0.01, lr_factor=0.1, lr_steps=[400000, 800000],
                end_epoch=1000, distributed=False, fine_tune=False,
                epoch_div_factor=4, precise_bn=False, args=None,
                **kwargs):

    assert torch.npu.is_available(), "Currently, we only support NPU version"

    # data iterator
    iter_seed = torch.initial_seed() \
                + (torch.distributed.get_rank() * 10 if distributed else 100) \
                + max(0, resume_epoch) * 100
    train_iter, eval_iter, train_sampler = iterator_factory.creat(name=dataset,
                                                   batch_size=batch_size,
                                                   clip_length=clip_length,
                                                   train_interval=train_frame_interval,
                                                   val_interval=val_frame_interval,
                                                   mean=input_conf['mean'],
                                                   std=input_conf['std'],
                                                   seed=iter_seed,
                                                   distributed=distributed)
    # model (dynamic)
    net = model(net=sym_net,
                criterion=torch.nn.CrossEntropyLoss().npu(),
                model_prefix=model_prefix,
                step_callback_freq=50,
                save_checkpoint_freq=save_frequency,
                opt_batch_size=batch_size, # optional
                single_checkpoint=precise_bn, # TODO: use shared filesystem to rsync running mean/var
                distributed=distributed,
                args=args
                )
    # if True:
    #     for name, module in net.net.named_modules():
    #         if name.endswith("bn"): module.momentum = 0.005
    if distributed:
        net.net = net.net.to(args.device)
        if args.master_node:
            logging.info("=================set args.device=============")
    else:
        net.net.npu()

    # config optimization, [[w/ wd], [w/o wd]]
    param_base_layers = [[[], []], [[], []]]
    param_new_layers = [[[], []], [[], []]]
    name_freeze_layers, name_base_layers = [], []
    for name, param in net.net.named_parameters():
        idx_wd = 0 if name.endswith('.bias') else 1
        idx_bn = 0 if name.endswith(('.bias', 'bn.weight')) else 1
        if fine_tune:
            if not name.startswith('classifier'):
                param_base_layers[idx_bn][idx_wd].append(param)
                name_base_layers.append(name)
            else:
                param_new_layers[idx_bn][idx_wd].append(param)
        else:
            if "conv_m2" in name:
                param_base_layers[idx_bn][idx_wd].append(param)
                name_base_layers.append(name)
            else:
                param_new_layers[idx_bn][idx_wd].append(param)


    if name_freeze_layers:
        out = "[\'" + '\', \''.join(name_freeze_layers) + "\']"
        if args.master_node:
            logging.info("Optimizer:: >> freezing {} params: {}".format(len(name_freeze_layers),
                         out if len(out) < 300 else out[0:150] + " ... " + out[-150:]))
    if name_base_layers:
        out = "[\'" + '\', \''.join(name_base_layers) + "\']"
        if args.master_node:
            logging.info("Optimizer:: >> recuding the learning rate of {} params: {}".format(len(name_base_layers),
                         out if len(out) < 300 else out[0:150] + " ... " + out[-150:]))

    if args.apex:
        optimizer = torch.optim.SGD(sym_net.parameters(), lr=lr_base, momentum=0.9)
        if args.master_node:
            logging.info(">>>using original SGD...")
    else:
        wd = 0.0001
        optimizer = custom_optim.SGD([{'params': param_base_layers[0][0], 'lr_mult': 0.5, 'weight_decay': 0.},
                                      {'params': param_base_layers[0][1], 'lr_mult': 0.5, 'weight_decay': wd},
                                      {'params': param_base_layers[1][0], 'lr_mult': 0.5, 'weight_decay': 0., 'name': 'precise.bn'},  # *.bias
                                      {'params': param_base_layers[1][1], 'lr_mult': 0.5, 'weight_decay': wd, 'name': 'precise.bn'},  # bn.weight
                                      {'params': param_new_layers[0][0],  'lr_mult': 1.0, 'weight_decay': 0.},
                                      {'params': param_new_layers[0][1],  'lr_mult': 1.0, 'weight_decay': wd}, 
                                      {'params': param_new_layers[1][0],  'lr_mult': 1.0, 'weight_decay': 0., 'name': 'precise.bn'},  # *.bias
                                      {'params': param_new_layers[1][1],  'lr_mult': 1.0, 'weight_decay': wd, 'name': 'precise.bn'}], # bn.weight
                                     lr=lr_base,
                                     momentum=0.9,
                                     nesterov=True)
        logging.info(">>>using self-define SGD...")

    if args.apex:
        from apex import amp
        net.net, optimizer = amp.initialize(net.net, optimizer, opt_level=args.apex_level, 
            loss_scale=args.loss_scale)
        if args.master_node:
            logging.info("======= Transfer Apex =======")
    elif args.master_node:
        logging.info("======= Don\'t Transfer Apex =======")

    net_without_ddp = net.net
    if distributed:
        local_rank = kwargs["local_rank"]
        net.net = torch.nn.parallel.DistributedDataParallel(net.net, device_ids=[local_rank],
                                output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False).npu()
        net.set_net_without_ddp(net_without_ddp)
        net.set_distributed_mode()
    elif args.gpus is not None:
        net.net = net.net.npu()

    # resume: model and optimizer
    if resume_epoch < 0:
        epoch_start = 0
        step_counter = 0
    else:
        net.load_checkpoint(epoch=resume_epoch, optimizer=optimizer)
        epoch_start = resume_epoch
        step_counter = epoch_start * int(train_iter.__len__()/epoch_div_factor)

    num_worker = torch.distributed.get_world_size() if distributed and torch.distributed.is_initialized else 1

    lr_scheduler = MultiFactorScheduler(base_lr=lr_base,
                                        steps=[int(x/(batch_size*num_worker)) for x in lr_steps],
                                        factor=lr_factor,
                                        step_counter=step_counter)

    metrics = metric.MetricList(metric.Loss(name="loss-ce"),
                                metric.Accuracy(name="top1", topk=1),
                                metric.Accuracy(name="top5", topk=5),)

    cudnn.benchmark = True

    net.fit(train_iter=train_iter,
            eval_iter=eval_iter,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            epoch_start=epoch_start,
            epoch_end=end_epoch,
            epoch_div_factor=epoch_div_factor,
            precise_bn=precise_bn,
            train_sampler=train_sampler)
