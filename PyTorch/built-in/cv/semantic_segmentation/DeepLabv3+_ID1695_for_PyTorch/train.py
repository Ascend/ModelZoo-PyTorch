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
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from apex import amp
import apex
import torch
if torch.__version__ >= '1.8':
    import torch_npu 
import time

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        kwargs['drop_last'] = False if args.multiprocessing_distributed else True
        self.train_loader, self.val_loader, self.test_loader, self.nclass, self.train_sampler = make_data_loader(args, **kwargs)
        self.nclass = args.num_classes if args.num_classes else self.nclass

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn,
                        device=args.device)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        if args.is_master_node:
            print(model)

        # Define Optimizer
        optimizer = apex.optimizers.NpuFusedSGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = nn.CrossEntropyLoss(size_average=True,ignore_index=255).to(args.device)
        self.model, self.optimizer = model, optimizer
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        self.model.to(args.device)
        if args.apex:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                        opt_level=args.apex_opt_level,
                                                        loss_scale = args.loss_scale_value,
                                                        combine_grad=True,
                                                        verbosity=1)

        model_dict = self.model.state_dict()
        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume,lambda storage, loc: storage)
            partial = {k.replace('module.',''):v for k, v in checkpoint['state_dict'].items() if k in model_dict}
            model_dict.update(partial)
            args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(model_dict,strict=False)
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            self.best_pred = 0
            args.start_epoch = 0

    def training(self, epoch):
        if self.args.multiprocessing_distributed:
            self.train_sampler.set_epoch(epoch)
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
          t_int = time.time()
          image, target = sample['image'], sample['label']
          image, target = image.to(self.args.device, non_blocking=True), target.to(self.args.device, non_blocking=True)
          self.scheduler(self.optimizer, i, epoch, self.best_pred)
          self.optimizer.zero_grad()
          output = self.model(image)
          output = output.permute(0, 2, 3, 1).reshape(-1,self.nclass)
          target = target.flatten().int()
          loss = self.criterion(output, target)
          if self.args.apex:
              with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                  scaled_loss.backward()
          else:
              loss.backward()
          self.optimizer.step()
          train_loss += loss.item()
          if i < 3 and epoch == 0:
              print("step_time: ", time.time() - t_int)
          if self.args.is_master_node:
              print('train_loss: %.3f' % (train_loss / (i + 1)))
              tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
              self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

              self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
              print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
          if self.args.no_val and self.args.is_master_node:
                # save checkpoint every epoch
                is_best = False
                self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
                }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            with torch.no_grad():
                output = self.model(image.to(self.args.device))
            output = output.permute(0, 2, 3, 1)
            output = torch.reshape(output, (-1,self.nclass))
            target = target.flatten().int()
            loss = self.criterion(output.to(self.args.device), target.to(self.args.device))
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        if self.args.is_master_node:
            self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
            self.writer.add_scalar('val/mIoU', mIoU, epoch)
            self.writer.add_scalar('val/Acc', Acc, epoch)
            self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
            self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
            print('Validation:')
            print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
            print("Acc:{}, Acc_class:{}, val_mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
            print('val_Loss: %.3f' % test_loss)

            new_pred = mIoU
            if new_pred > self.best_pred:
                is_best = True
                self.best_pred = new_pred
                self.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                }, is_best)
            print("Current best mIoU: {}".format(self.best_pred))

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--num_classes', type=int, default=0,
                        help='dataset num classes')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default='',
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    # mix precision training
    parser.add_argument('--apex', action='store_true',
                        help='User apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O2', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precison training.')
    parser.add_argument('--loss-scale-value', default=1024,
                        help='loss scale using in amp, default -1 means dynamic')
    parser.add_argument('--multiprocessing_distributed', action='store_true',
                        help='Use multi-processing distributed training to'
                             'launch N processes per node, which has N NPUs.'
                             'This is the fastest way to use PyTorch for'
                             'either single node or multi node data parallel'
                             'training')
    parser.add_argument('--device_id', default=5, type=int, help='device id')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29688'

    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)

    if args.multiprocessing_distributed:
        args.rank_id = int(os.environ['RANK_ID'])
        args.device_id = args.rank_id
        args.world_size = int(os.environ['RANK_SIZE'])
        torch.distributed.init_process_group(backend='hccl', init_method='env://',
                            world_size=args.world_size, rank=args.rank_id)

    args.is_master_node = not args.multiprocessing_distributed or args.device_id == 0
    if args.is_master_node:
        print(args)

    args.device = torch.device(f'npu:{args.device_id}')
    torch.npu.set_device(args.device)

    if args.multiprocessing_distributed:
        args.batch_size = int(args.batch_size / args.world_size)
        args.workers = int((args.workers + args.world_size - 1) / args.world_size)

    trainer = Trainer(args)
    if args.multiprocessing_distributed:
        trainer.model = torch.nn.parallel.DistributedDataParallel(trainer.model,
                                        device_ids=[args.device_id])

    if args.is_master_node:
        print('Starting Epoch:', trainer.args.start_epoch)
        print('Total Epoches:', trainer.args.epochs)

    used_time = []
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        t_s = time.time()
        trainer.training(epoch)
        if args.is_master_node and epoch >= 2:
            used_time.append(time.time() - t_s)
            avg_time = np.average(used_time)
            print("Epoch {}: average s/epoch: {} FPS: {}".format(epoch, avg_time, 1464/avg_time))
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()
