"""
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the BSD 3-Clause License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://spdx.org/licenses/BSD-3-Clause.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
from reid.trainers_partloss import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint

from apex import amp


import os

def get_data(name, data_dir, height, width, batch_size, workers, device_num):
    root = osp.join(data_dir, name)
    root = data_dir
    dataset = datasets.create(name, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    num_classes = dataset.num_train_ids

    train_transformer = T.Compose([
        T.RectScale(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    train_sampler = None
    if device_num > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset.train)
        train_loader = DataLoader(
            Preprocessor(dataset.train, root=osp.join(dataset.images_dir,dataset.train_path),
                        transform=train_transformer),
            batch_size=batch_size, num_workers=workers,
            shuffle=False, pin_memory=True, drop_last=True, sampler=train_sampler)
    else:
        train_loader = DataLoader(
            Preprocessor(dataset.train, root=osp.join(dataset.images_dir,dataset.train_path),
                        transform=train_transformer),
            batch_size=batch_size, num_workers=workers,
            shuffle=True, pin_memory=True, drop_last=True)


    query_loader = DataLoader(
        Preprocessor(dataset.query, root=osp.join(dataset.images_dir,dataset.query_path),
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, root=osp.join(dataset.images_dir,dataset.gallery_path),
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, query_loader, gallery_loader, train_sampler


def  main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device_num == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
    else:
        environ_str = '0'
        for i in range(1, args.device_num):
            environ_str = environ_str + ',%d' % i
        os.environ["CUDA_VISIBLE_DEVICES"] = environ_str
            

    if args.npu:
        os.environ['MASTER_ADDR'] = args.addr
        os.environ['MASTER_PORT'] = '29688'
        if args.device_num > 1:
            torch.distributed.init_process_group(backend="hccl", rank=args.local_rank, world_size=args.device_num)
        torch.npu.manual_seed_all(args.seed)
        torch.npu.set_device(args.local_rank)
        os.environ['device'] = 'npu'

    else:
        if args.device_num > 1:
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = True
        torch.cuda.set_device(args.local_rank)
        os.environ['device'] = 'gpu'


    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)
    dataset, num_classes, train_loader, query_loader, gallery_loader, train_sampler = \
        get_data(args.dataset,  args.data_dir, args.height,
                 args.width, args.batch_size, args.workers, args.device_num
                 )


    # Create model
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes,cut_at_pooling=False, FCN=True)

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model_dict = model.state_dict()
        checkpoint_load = {k: v for k, v in (checkpoint['state_dict']).items() if k in model_dict}
        model_dict.update(checkpoint_load)
        model.load_state_dict(model_dict)
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        amp.load_state_dict(checkpoint['amp'])
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))
        
    # Optimizer
    if hasattr(model, 'base'):
        base_param_ids = set(map(id, model.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': model.base.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
    else:
        param_groups = model.parameters()

    if args.npu:
        from apex.optimizers import NpuFusedSGD
        optimizer = NpuFusedSGD(param_groups, lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay,
                            nesterov=True)
        model = model.npu()
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=128.0, combine_grad=True)
    else:
        optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
        model = model.cuda()
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=128.0)

    if args.device_num > 1:
        model = nn.parallel.DistributedDataParallel(model, 
                    device_ids=[args.local_rank], 
                    output_device=args.local_rank,
                    find_unused_parameters=True,
                    broadcast_buffers=False
                    )
    else:
        model = nn.DataParallel(model)

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        print("Test:")
        checkpoint = load_checkpoint(osp.join(args.logs_dir, 'checkpoint.pth.tar'))
        model.module.load_state_dict(checkpoint['state_dict'])
        amp.load_state_dict(checkpoint['amp'])
        evaluator.evaluate(query_loader, gallery_loader,  dataset.query, dataset.gallery)
        return

    # Criterion
    if args.npu:
        criterion = nn.CrossEntropyLoss().npu()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    # Trainer
    trainer = Trainer(model, criterion, 0, 0, SMLoss_mode=0)

    # Schedule learning rate
    def adjust_lr(epoch):
        step_size = 60 if args.arch == 'inception' else args.step_size
        lr = args.lr * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    total_avg = 0.0
    # Start training
    for epoch in range(start_epoch, args.epochs):
        if args.device_num > 1:
            train_sampler.set_epoch(epoch)
        adjust_lr(epoch)
        use_time = trainer.train(epoch, train_loader, optimizer)
        total_avg += use_time
        is_best = True
        
        if args.local_rank == 0:
            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch': epoch + 1,
                'best_top1': best_top1,
                'amp': amp.state_dict(),
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
    avg_time = total_avg / (args.epochs - start_epoch)

    if not args.performance:
        print('Test with best model:')
        evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery)

    device_num = 1 if args.device_num == -1 else args.device_num
    print('FPS@all {:.3f}, TIME@all {:.3f}'.format(device_num * args.batch_size / avg_time, avg_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--step-size',type=int, default=40)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))

    # 多卡
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--device_num', default=-1, type=int,
                        help='device_num')
    parser.add_argument('--npu', action='store_true',
                        help="npu")
    parser.add_argument('--addr', default='127.0.0.1',
                    type=str, help='master addr')
    parser.add_argument('--performance', action='store_true',
                    help="performance")
    main(parser.parse_args())
