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
import json
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from run.opts import parse_opts

from run.spatial_transforms import *
from run.temporal_transforms import *
from run.target_transforms import ClassLabel, VideoID
from run.dataset import get_train_loader, get_val_loader, get_test_loader
from run.train import train_epoch
from run.utils import adjust_learning_rate, save_checkpoint, Logger, AverageMeter, opt_preprocess
from run.validation import val_epoch
from run.test import test
# import test
from run.getmodel import generate_model, model_load_pretrained, replacemudule
import torch.multiprocessing as mp

from test_acc import final_test

if __name__ == '__main__':
    opt = parse_opts()
    opt = opt_preprocess(opt)

    if torch.cuda.is_available():
        opt.gpu_or_npu = 'gpu'
        # print('gpu...')
    else:
        opt.gpu_or_npu = 'npu'
        # print('npu...')

    torch.manual_seed(opt.manual_seed)

    if not opt.no_drive:
        if opt.gpu_or_npu == 'gpu':
            torch.cuda.manual_seed(opt.manual_seed)
            # set device states
            os.environ["CUDA_VISIBLE_DEVICES"] = opt.device_lists  # before using torch

            if opt.distributed:
                import torch.distributed as dist
                from torch.nn.parallel import DistributedDataParallel
                torch.cuda.set_device(opt.local_rank)
                dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:50000', world_size=opt.world_size, rank=opt.local_rank)
                print('Initializing process rank:', opt.local_rank, '\t with gpus:', opt.device_lists)
                opt.device = torch.device("cuda", opt.local_rank)
                print(opt.device)
                torch.cuda.set_device(opt.local_rank)
            else:
                opt.device = torch.device("cuda:0")
                torch.cuda.set_device(opt.device)

        elif opt.gpu_or_npu == 'npu':
            torch.npu.manual_seed(opt.manual_seed)
            import torch.npu
            import torch.backends.cudnn as cudnn

            if opt.distributed:
                import torch.distributed as dist
                from torch.nn.parallel import DistributedDataParallel
                os.environ['MASTER_ADDR'] = opt.addr
                os.environ['MASTER_PORT'] = '50000'
                dist.init_process_group(backend='hccl', world_size=opt.world_size, rank=opt.local_rank)
                print('Initializing process rank:', opt.local_rank, '\t with npus', opt.device_lists)
                opt.device = torch.device("npu:{}".format(opt.local_rank))
                torch.npu.set_device(opt.device)
            else:
                opt.device = torch.device("npu:0")
                torch.npu.set_device(opt.device)

    if (opt.distributed and dist.get_rank() == 0) or (opt.device_num == 1):  # distributed master or 1p
        print(opt)

    # get model
    model = generate_model(opt)
    model, parameters = model_load_pretrained(opt, model)

    if not opt.no_drive:
        model = model.to(opt.device)

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening

    if opt.gpu_or_npu == 'gpu':
        optimizer = optim.SGD(parameters, lr=opt.learning_rate, momentum=opt.momentum, dampening=dampening, weight_decay=opt.weight_decay, nesterov=opt.nesterov)
    elif opt.gpu_or_npu == 'npu':
        if opt.use_apex == 1:
            import apex
            optimizer = apex.optimizers.NpuFusedAdam(parameters, lr=opt.learning_rate)
        else:
            optimizer = optim.Adam(parameters, lr=opt.learning_rate)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)

    if opt.use_apex == 1:
        from apex import amp
        if opt.loss_scale < 0:
            model, optimizer = amp.initialize(model, optimizer, opt_level=opt.opt_level, loss_scale=None)
        else:
            # model, optimizer = amp.initialize(model, optimizer, opt_level=opt.opt_level, loss_scale=opt.loss_scale, combine_grad=True)
            model, optimizer = amp.initialize(model, optimizer, opt_level=opt.opt_level, loss_scale=opt.loss_scale)

    # resume model
    best_pre1 = 0
    if opt.resume_path:
        if (opt.distributed and dist.get_rank() == 0) or (opt.device_num == 1):  # distributed master or 1p
            print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']
        best_pre1 = checkpoint['best_prec1']
        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(replacemudule(checkpoint['state_dict']))
        if opt.use_apex == 1:
            amp.load_state_dict(checkpoint['amp'])

    if (opt.distributed and dist.get_rank() == 0) or (opt.device_num == 1):  # distributed master or 1p
        print('run')

    if opt.gpu_or_npu == 'gpu' and opt.distributed:
        model = DistributedDataParallel(model, device_ids=[opt.local_rank])
    elif opt.gpu_or_npu == 'npu' and opt.distributed:
        model = DistributedDataParallel(model, device_ids=[opt.local_rank], broadcast_buffers=False)

    criterion = nn.CrossEntropyLoss()
    if not opt.no_drive:
        criterion = criterion.to(opt.device)

    # train data
    train_loader = None
    if not opt.no_train:
        train_loader = get_train_loader(opt)
        train_logger = Logger(os.path.join(opt.result_path, 'train.log'),
                              ['date', 'epoch', 'fps', 'loss', 'prec1', 'prec5', 'lr'])
        train_batch_logger = Logger(os.path.join(opt.result_path, 'train_batch.log'),
                                    ['date', 'epoch', 'batch', 'iter', 'fps', 'loss', 'prec1', 'prec5', 'lr'])

    # valid data
    val_loader = None
    if not opt.no_val:
        val_loader = get_val_loader(opt)
        val_logger = Logger(os.path.join(opt.result_path, 'val.log'), ['date', 'epoch', 'loss', 'prec1', 'prec5'])

    # train and valid
    if not opt.no_train or not opt.no_val:
        for i in range(opt.begin_epoch, opt.n_epochs + 1):

            # train set
            if not opt.no_train:
                if opt.distributed:
                    train_loader.sampler.set_epoch(i)

                adjust_learning_rate(optimizer, i, opt)
                if opt.distributed:
                    train_epoch(i, train_loader, model, criterion, optimizer, opt, train_logger, train_batch_logger, device_ids=opt.local_rank)
                else:
                    train_epoch(i, train_loader, model, criterion, optimizer, opt, train_logger, train_batch_logger, device_ids=0)

                if (opt.distributed and dist.get_rank() == 0) or (opt.device_num == 1):  # distributed master or 1p
                    state = {
                        'epoch': i,
                        'arch': opt.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_prec1': best_pre1
                    }
                    if opt.use_apex == 1:
                        state = {
                            'epoch': i,
                            'arch': opt.arch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_prec1': best_pre1,
                            'amp': amp.state_dict()
                        }
                    save_checkpoint(state, False, opt)

            # valid set
            if not opt.no_val:
                if opt.distributed:
                    validation_loss, pre1 = val_epoch(i, val_loader, model, criterion, opt, val_logger, device_ids=opt.local_rank)
                else:
                    validation_loss, pre1 = val_epoch(i, val_loader, model, criterion, opt, val_logger, device_ids=0)

                if (opt.distributed and dist.get_rank() == 0) or (opt.device_num == 1):  # distributed master or 1p
                    is_best = pre1 > best_pre1
                    best_pre1 = max(pre1, best_pre1)

                    state = {
                        'epoch': i,
                        'arch': opt.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_prec1': best_pre1
                        }
                    if opt.use_apex == 1:
                        state = {
                            'epoch': i,
                            'arch': opt.arch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_prec1': best_pre1,
                            'amp': amp.state_dict()
                        }
                    save_checkpoint(state, is_best, opt)



    if opt.test:
        # get model
        del model, optimizer, train_loader, val_loader
        if opt.gpu_or_npu == 'gpu':
            torch.cuda.empty_cache()
        elif opt.gpu_or_npu == 'npu':
            torch.npu.empty_cache()
        if (opt.distributed and dist.get_rank() == 0) or (opt.device_num == 1):  # distributed master or 1p
            final_test(opt)
            print('train finish!')



