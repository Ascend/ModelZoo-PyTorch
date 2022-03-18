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
# Copyright (c) Runpei Dong, ArChip Lab.
# 
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

import torch

import argparse
import time
import os
import sys
import math

import torch.nn as nn
import config as cfg

from tqdm import tqdm
from mypath import Path
from dataloader import make_data_loader
from modeling import DGMSNet
from modeling.DGMS import DGMSConv
from utils.sparsity import SparsityMeasure
from utils.lr_scheduler import get_scheduler
from utils.PyTransformer.transformers.torchTransformer import TorchTransformer
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.saver import Saver
from utils.misc import AverageMeter, get_optimizer, resume_ckpt
from utils.loss import *
import torch.npu
import os
import apex 
from apex import amp
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

class Trainer(object):
    def __init__(self, args):
        self.args = args
        cfg.set_config(args)

        self.saver = Saver(args)
        self.saver.save_experiment_config()

        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        model = DGMSNet(args, args.freeze_bn)
        
        if args.mask:
            print("DGMS Conv!")
            _transformer = TorchTransformer()
            _transformer.register(nn.Conv2d, DGMSConv)
            model = _transformer.trans_layers(model)
        else:
            print("Normal Conv!")

        print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

        cfg.IS_NORMAL = True if (args.resume is not None) else False
        optimizer = get_optimizer(model, args)
        cfg.IS_NORMAL = self.args.normal
        self.model, self.optimizer = model, optimizer
        self.model.npu()
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                          opt_level="O2",
                                          loss_scale=128,
                                          combine_grad=True
                                          )

        self.criterion = nn.CrossEntropyLoss()
        self.sparsity = SparsityMeasure(args)

        self.lr_scheduler = get_scheduler(args, self.optimizer, args.lr, len(self.train_loader))
       
        self.evaluator = Evaluator(self.nclass, self.args)
        
        if args.cuda:
            torch.backends.cudnn.benchmark=True
            self.model = torch.nn.parallel.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = self.model.npu()

        self.best_top1 = 0.0
        self.best_top5 = 0.0
        self.best_sparse_ratio = 0.0
        self.this_sparsity = 0.0
        self.best_params = 0.0
        if args.resume is not None:
            self.model, self.optimizer, self.lr_scheduler, self.best_top1 = \
                resume_ckpt(args, self.model, self.train_loader, self.optimizer, self.lr_scheduler)

        print('    Total params (+GMM) : %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        
        if args.rt:
            args.start_epoch = 0

    def training(self, epoch):
        cfg.set_status(True)
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        train_loss = 0.0
        num_img_tr = len(self.train_loader)

        tbar = tqdm(self.train_loader)
        for i, (image, target) in enumerate(tbar):
            if self.args.max_steps and i > self.args.max_steps:
                break
            start = time.time()
            data_time.update(time.time() - end)
            if self.args.cuda:
                image, target = image.npu(), target.npu()
            outputs = self.model(image)
            loss = self.criterion(outputs, target)

            prec1, prec5 = self.evaluator.Accuracy(outputs.data, target.data, topk=(1, 5))
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))
            top5.update(prec5.item(), image.size(0))

            self.optimizer.zero_grad()

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=True)

            #loss.backward(retain_graph=True)

            self.optimizer.step()
            self.lr_scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

            fps = self.args.batch_size / (time.time() - start)

            train_loss = (loss.item() + train_loss)
            tbar.set_description('Train Loss: {loss:.4f} | T1: {top1: .3f} | T5: {top5: .2f} | best T1: {pre_best:.2f} T5: {best_top5:.2f} NZ: {nz_val:.4f} #Params: {params:.2f}M | lr: {_lr:.8f} | fps: {fps:.2f}'
                .format(loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        pre_best=self.best_top1,
                        best_top5=self.best_top5,
                        nz_val=1-self.best_sparse_ratio,
                        params=self.best_params,
                        _lr=self.optimizer.param_groups[0]['lr'],
                        fps=fps
                        ))
            self.writer.add_scalar('train/train_loss_iter', loss.item(), i + num_img_tr * epoch)
            self.writer.add_scalar('train/top1', top1.avg, i + num_img_tr * epoch)
            self.writer.add_scalar('train/top5', top5.avg, i + num_img_tr * epoch)
            self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], i + num_img_tr * epoch)
            
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        return (losses.avg, top1.avg)

    def validation(self, epoch):
        cfg.set_status(False)
        num_img_tr = len(self.val_loader)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        end = time.time()
        for i, (image, target) in enumerate(tbar):
            data_time.update(time.time() - end)

            if self.args.cuda:
                image, target = image.npu(), target.npu()
            with torch.no_grad():
                outputs = self.model(image)
            loss = self.criterion(outputs, target)
            test_loss += loss.item()

            prec1, prec5 = self.evaluator.Accuracy(outputs.data, target.data, topk=(1, 5))
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))
            top5.update(prec5.item(), image.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            tbar.set_description('({batch}/{size}) Test Loss: {loss:.4f} | Top1: {top1: .4f} | Top5: {top5: .4f}'
                .format(batch=i + 1,
                        size=len(self.val_loader),
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        ))
            self.writer.add_scalar('val/val_loss_iter', loss.item(), i + num_img_tr * epoch)
            self.writer.add_scalar('val/top1', top1.avg, i + num_img_tr * epoch)
            self.writer.add_scalar('val/top5', top5.avg, i + num_img_tr * epoch)
        if self.args.show_info:
            self.this_sparsity, this_params = self.sparsity.check_sparsity_per_layer(self.model)
            self.writer.add_scalar('val/total_sparsity', self.this_sparsity, epoch)
        new_pred = top1.avg
        if new_pred > self.best_top1 and not self.args.only_inference:
            is_best = True
            self.best_top1 = new_pred
            self.best_params = this_params
            self.best_top5 = top5.avg
            self.best_sparse_ratio = self.this_sparsity
            bitwidth = math.floor(math.log(cfg.K_LEVEL, 2))
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_top1': self.best_top1,
                'best_top5': self.best_top5,
                'params': self.best_params,
                'bits': bitwidth,
                'CR': 1/((1-self.best_sparse_ratio) * bitwidth / 32),
                'according_sparsity': self.this_sparsity,
            }, is_best)
        return (losses.avg, top1.avg)

def main():
    parser = argparse.ArgumentParser(description="Differentiable Gaussian Mixture Weight Sharing (DGMS)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--network', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50', 'mnasnet', 'proxylessnas',
                                 'resnet20', 'resnet32', 'resnet56', 'vggsmall'],
                        help='network name (default: resnet18)')
    parser.add_argument('-d', '--dataset', type=str, default='imagenet',
                        choices=['cifar10', 'imagenet', 'cars', 'cub200', 'aircraft'],
                        help='dataset name (default: imgenet)')
    parser.add_argument('-j', '--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=32,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=32,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--train-dir', type=str, default=None,
                        help='training set directory (default: None)')
    parser.add_argument('--val-dir', type=str, default='None',
                        help='validation set directory (default: None)')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='Number of classes (default: 1000)')
    parser.add_argument('--show-info', action='store_true', default=False, 
                        help='set if show model compression info (default: False)')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N', 
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N', 
                        help='input batch size for testing (default: 256)')
    # model params
    parser.add_argument('--K', type=int, default=16, metavar='K',
                        help='number of GMM components (default: 2^4=16)')
    parser.add_argument('--tau', type=float, default=0.01, metavar='TAU',
                        help='gumbel softmax temperature (default: 0.01)')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='whether train noramlly (default: False)')
    parser.add_argument('--empirical', type=bool, default=False,
                        help='whether use empirical initialization for parameter sigma (default: False)')
    parser.add_argument('--mask', action='store_true', default=False,
                        help='whether transform normal convolution into DGMS convolution (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=2e-5, metavar='LR',
                        help='learning rate (default: 2e-5)')
    parser.add_argument('--lr-scheduler', type=str, default='one-cycle',
                        choices=['one-cycle', 'cosine', 'multi-step', 'reduce'],
                        help='lr scheduler mode: (default: one-cycle)')
    parser.add_argument('--schedule', type=str, default='70,140,190')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training (default: False)')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default="Experiments",
                        help='set the checkpoint name')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='set if use a pretrained network')
    # re-train a pre-trained model
    parser.add_argument('--rt', action='store_true', default=False,
                        help='retraining model for quantization')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--only-inference', type=bool, default=False,
                        help='skip training and only inference')
    parser.add_argument('--max_steps', default=None, type=int, metavar='N',
                        help='number of total steps to run')
    # loss_scale
    parser.add_argument('--loss_scale', default=128, type=float, help='loss scale using in amp, default -1 means dynamic')

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )


    args = parser.parse_args()
    args.schedule = [int(s) for s in args.schedule.split(',')]
    args.cuda = not args.no_cuda and torch.npu.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError("Argument --gpu_ids must be a comma-separeted list of integers only")
    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    if args.epochs is None:
        args.epochs = cfg.EPOCH[args.dataset.lower()]
    
    if args.num_classes is None:
        args.num_classes = cfg.NUM_CLASSES[args.dataset.lower()]
    
    if args.train_dir is None or args.val_dir is None:
        args.train_dir, args.val_dir = Path.db_root_dir(args.dataset.lower()), Path.db_root_dir(args.dataset.lower())

    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    if args.only_inference:
        print("Only inference with given resumed model...")
        val_loss, val_acc = trainer.validation(0)
        return
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        train_loss, train_acc = trainer.training(epoch)
        if epoch % args.eval_interval == (args.eval_interval - 1):
            val_loss, val_acc = trainer.validation(epoch)
    nz_val = 1 - trainer.best_sparse_ratio
    params_val = trainer.best_params
    compression_rate = 1/(nz_val * (math.floor(math.log(cfg.K_LEVEL, 2)) / 32))
    print(f"Best Top-1: {trainer.best_top1} | Top-5: {trainer.best_top5} | NZ: {nz_val} | #Params: {params_val:.2f}M | CR: {compression_rate:.2f}")

    trainer.writer.close()

if __name__ == '__main__':
    main()
