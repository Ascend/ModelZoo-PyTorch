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

import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler
from util.shedule import FixLR

from dataset.total_text import TotalText
from dataset.synth_text import SynthText
from network.loss import TextLoss
from network.textnet import TextNet
from util.augmentation import BaseTransform, Augmentation
from util.config import config as cfg, update_config, print_config
from util.misc import AverageMeter
from util.misc import mkdirs, to_device,to_device_parrall
from util.option import BaseOptions
from util.visualize import visualize_network_output
from util.summary import LogSummary

import torch.multiprocessing as mp
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
import apex
import random



lr = None
train_step = 0

def save_model(model, epoch, lr, optimzer,cfg):
    save_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)

    #save_path = os.path.join(save_dir, 'textsnake_{}_{}.pth'.format(model.backbone_name, epoch))
    save_path = os.path.join(save_dir, 'textsnake_{}.pth'.format(epoch))
    print('Saving to {}.'.format(save_path))
    state_dict = {
        'lr': lr,
        'epoch': epoch,
        'model': model.state_dict() if not cfg.mgpu else model.module.state_dict(),
        'optimizer': optimzer.state_dict()
    }
    torch.save(state_dict, save_path)


def load_model(model, model_path):
    print('Loading from {}'.format(model_path))
    state_dict = torch.load(model_path, lambda storage, loc: storage)
    model.load_state_dict({k.replace('module.',''):v for k,v in state_dict['model'].items()})


def train(model, train_loader, criterion, scheduler, optimizer, epoch, gpu, cfg):

    global train_step

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #top1 = AverageMeter('Acc@1', ':6.2f')
    #top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    #list_loss = []
    scheduler.step()

    print('Epoch: {} : LR = {}'.format(epoch, scheduler.get_lr()))
    loc = 'npu:{}'.format(gpu)

    for i, (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        train_step += 1
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = to_device_parrall(gpu,
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map)
        # backward
        
        
        output = model(img)
        
        criterion = TextLoss().cpu()
        output = output.cpu()
        tr_mask = tr_mask.cpu()
        tcl_mask = tcl_mask.cpu()
        sin_map = sin_map.cpu()
        cos_map = cos_map.cpu()
        radius_map = radius_map.cpu()
        train_mask = train_mask.cpu()
        

        tr_loss, tcl_loss, sin_loss, cos_loss, radii_loss = \
            criterion(output, tr_mask, tcl_mask, sin_map, cos_map, radius_map, train_mask)
        loss = tr_loss + tcl_loss + sin_loss + cos_loss + radii_loss
        loss = loss.to(loc)

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()
        optimizer.step()

        
        losses.update(loss.item())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        FPS = cfg.batch_size * cfg.world_size / batch_time.avg

        #if i % 10==0:
        #    print('FPS: {:.3f}'.format(FPS))

        if cfg.viz and i % cfg.viz_freq == 0 and gpu==0:
            visualize_network_output(output, tr_mask, tcl_mask, mode='train', cfg=cfg)

        if i % cfg.display_freq == 0:
            print('({:d} / {:d}) - Loss: {:.4f} - tr_loss: {:.4f} - tcl_loss: {:.4f} - sin_loss: {:.4f} - cos_loss: {:.4f} - radii_loss: {:.4f}'.format(
                i, len(train_loader), loss.item(), tr_loss.item(), tcl_loss.item(), sin_loss.item(), cos_loss.item(), radii_loss.item())
            )

        #if i % cfg.log_freq == 0:
        #    logger.write_scalars({
        #        'loss': loss.item(),
        #        'tr_loss': tr_loss.item(),
        #        'tcl_loss': tcl_loss.item(),
        #        'sin_loss': sin_loss.item(),
        #        'cos_loss': cos_loss.item(),
        #        'radii_loss': radii_loss.item()
        #    }, tag='train', n_iter=train_step)

    if epoch % cfg.save_freq == 0:
        #if i % 10000 == 0:
        if(gpu==0):
            save_model(model, epoch, scheduler.get_lr(), optimizer,cfg)
        
    FPS = cfg.batch_size * cfg.world_size / batch_time.avg
    print('Training Loss: {}'.format(losses.avg))
    print('FPS: {:.3f}'.format(FPS))


def validation(model, valid_loader, criterion, epoch,gpu,cfg):
    with torch.no_grad():
        model.eval()
        losses = AverageMeter()
        tr_losses = AverageMeter()
        tcl_losses = AverageMeter()
        sin_losses = AverageMeter()
        cos_losses = AverageMeter()
        radii_losses = AverageMeter()

        for i, (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta) in enumerate(valid_loader):

            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = to_device_parrall(gpu,
                img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map)

            output = model(img)

            tr_loss, tcl_loss, sin_loss, cos_loss, radii_loss = \
                criterion(output, tr_mask, tcl_mask, sin_map, cos_map, radius_map, train_mask)
            loss = tr_loss + tcl_loss + sin_loss + cos_loss + radii_loss

            # update losses
            losses.update(loss.item())
            tr_losses.update(tr_loss.item())
            tcl_losses.update(tcl_loss.item())
            sin_losses.update(sin_loss.item())
            cos_losses.update(cos_loss.item())
            radii_losses.update(radii_loss.item())

            if cfg.viz and i % cfg.viz_freq == 0:
                visualize_network_output(output, tr_mask, tcl_mask, mode='val',cfg=cfg)

            if i % cfg.display_freq == 0:
                print(
                    'Validation: - Loss: {:.4f} - tr_loss: {:.4f} - tcl_loss: {:.4f} - sin_loss: {:.4f} - cos_loss: {:.4f} - radii_loss: {:.4f}'.format(
                        loss.item(), tr_loss.item(), tcl_loss.item(), sin_loss.item(),
                        cos_loss.item(), radii_loss.item())
                )

        #logger.write_scalars({
        #    'loss': losses.avg,
        #    'tr_loss': tr_losses.avg,
        #    'tcl_loss': tcl_losses.avg,
        #    'sin_loss': sin_losses.avg,
        #    'cos_loss': cos_losses.avg,
        #    'radii_loss': radii_losses.avg
        #}, tag='val', n_iter=epoch)

        print('Validation Loss: {}'.format(losses.avg))

def device_id_to_process_device_map(device_list):

    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map

def main():

    global lr
    
    #########################################################
    #cfg.world_size = cfg.gpus * cfg.nodes                #
    os.environ['MASTER_ADDR'] = 'localhost'              #
    os.environ['MASTER_PORT'] = str(random.randrange(1001,49999))                      #
    #mp.spawn(train_pre, nprocs=cfg.gpus, args=(cfg,))         #

    if cfg.dist_url == "env://" and cfg.world_size == -1:
        cfg.world_size = int(os.environ["WORLD_SIZE"])

    cfg.distributed = cfg.world_size > 1 or cfg.multiprocessing_distributed

    #cfg.process_device_map = device_id_to_process_device_map(cfg.device_list)
    #cfg.process_device_map = {"0":0,"1":1}



    if cfg.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        #cfg.world_size = ngpus_per_node * cfg.world_size
        cfg.world_size = cfg.gpus * cfg.nodes
        print(cfg.world_size)
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(train_pre, nprocs=cfg.gpus,
                 args=(cfg.gpus, cfg))
    else:
        # Simply call main_worker function
        cfg.world_size = 1
        train_pre(cfg.gpu, 1, cfg)
    #########################################################


def train_pre(gpu,ngpus_per_node, cfg):
    #option = BaseOptions()
    #args = option.initialize()

    #update_config(cfg, args)
    global lr
    #print( cfg.process_device_map)
    #cfg.gpu = cfg.process_device_map[str(gpu)]
    rank = gpu
    loc = 'npu:{}'.format(rank)
    torch.npu.set_device(loc)


    if cfg.distributed:
        torch.npu.set_device(rank)
        dist.init_process_group(backend='hccl',   init_method="env://",
                                    world_size=cfg.world_size, rank=rank)
                                                     
    ############################################################
    

    if cfg.dataset == 'total-text':

        trainset = TotalText(
            data_root='data/total-text',
            ignore_list=None,
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )

        valset = TotalText(
            data_root='data/total-text',
            ignore_list=None,
            is_training=False,
            transform=BaseTransform(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )

    elif cfg.dataset == 'synth-text':
        trainset = SynthText(
            data_root='data/SynthText',
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None
    else:
        pass

     #train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    if cfg.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset,num_replicas=cfg.world_size,rank=rank)
    else:
        train_sampler = None
    train_loader = data.DataLoader(dataset=trainset,    
                                pin_memory=False,
                                shuffle=(train_sampler is None),
                                batch_size=cfg.batch_size, 
                                num_workers=cfg.num_workers,
                                sampler=train_sampler)
    if valset:
        if cfg.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(valset,num_replicas=cfg.world_size,rank=rank)
        else:
            val_sampler = None
        val_loader = data.DataLoader(dataset=valset,    
                                pin_memory=False,
                                shuffle=(val_sampler is None),
                                batch_size=cfg.batch_size, 
                                num_workers=cfg.num_workers,
                                sampler=val_sampler)
        #val_loader = data.DataLoader(valset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    else:
        valset = None

    #log_dir = os.path.join(cfg.log_dir, datetime.now().strftime('%b%d_%H-%M-%S_') + cfg.exp_name)
    #logger = LogSummary(log_dir)

    # Model
    model = TextNet(is_training=True, backbone=cfg.net)
    model.to(loc)
    if cfg.resume:
        load_model(model, cfg.resume)
    lr = cfg.lr
    
    optimizer = apex.optimizers.NpuFusedAdam(model.parameters(), lr=cfg.lr)
    #optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale=None,combine_grad=True)

    if cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False)

    criterion = TextLoss().to(loc)


    if cfg.dataset == 'synth-text':
        scheduler = FixLR(optimizer)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    print('Start training TextSnake.')
    for epoch in range(cfg.start_epoch, cfg.max_epoch):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
        train(model, train_loader, criterion, scheduler, optimizer, epoch, rank, cfg)
        if valset:
            validation(model, val_loader, criterion, epoch,rank,cfg)

    print('End.')


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == "__main__":
    # parse arguments
    #torch.multiprocessing.set_start_method('spawn')
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    # main
    main()
