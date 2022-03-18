# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-

import argparse
import time
import datetime
import os
import shutil
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

CALCULATE_DEVICE = "npu:0"

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import torch.npu

from torchvision import transforms
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.loss import get_segmentation_loss
from core.utils.distributed import *
from core.utils.logger import setup_logger
from core.utils.lr_scheduler import WarmupPolyLR
from core.utils.score import SegmentationMetric
from apex import amp


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='fcn',
                        choices=['fcn32s', 'fcn16s', 'fcn8s', 'fcn', 'psp', 'deeplabv3',
                            'deeplabv3_plus', 'danet', 'denseaspp', 'bisenet', 'encnet',
                            'dunet', 'icnet', 'enet', 'ocnet', 'psanet', 'cgnet', 'espnet',
                            'lednet', 'dfanet'],
                        help='model name (default: fcn32s)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['vgg16', 'resnet18', 'resnet50', 'resnet101', 'resnet152',
                            'densenet121', 'densenet161', 'densenet169', 'densenet201'],
                        help='backbone name (default: vgg16)')
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        choices=['pascal_voc', 'pascal_aug', 'ade20k', 'citys', 'sbu'],
                        help='dataset name (default: pascal_voc)')
    parser.add_argument('--base-size', type=int, default=520,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
    parser.add_argument('--workers', '-j', type=int, default=4,
                        metavar='N', help='dataloader threads')
    # training hyper params
    parser.add_argument('--jpu', action='store_true', default=False,
                        help='JPU')
    parser.add_argument('--use-ohem', type=bool, default=False,
                        help='OHEM Loss for cityscapes dataset')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--warmup-iters', type=int, default=0,
                        help='warmup iters')
    parser.add_argument('--warmup-factor', type=float, default=1.0 / 3,
                        help='lr = warmup_factor * lr')
    parser.add_argument('--warmup-method', type=str, default='linear',
                        help='method of warmup')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='~/.torch/models',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='../runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    # apex
    parser.add_argument('--amp', default=False, action='store_true',
                        help='use amp to train the model')
    parser.add_argument('--loss-scale', default=128.0, type=float,
                        help='loss scale using in amp, default -1 means dynamic')
    parser.add_argument('--opt-level', default='O2', type=str,
                        help='loss scale using in amp, default -1 means dynamic')

    # npu setting
    parser.add_argument('--device', default='npu', type=str,
                        help='npu or gpu')
    parser.add_argument('--dist-backend', default='hccl', type=str,
                        help='distributed backend')

    args = parser.parse_args()

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'pascal_aug': 80,
            'pascal_voc': 50,
            'pcontext': 80,
            'ade20k': 160,
            'citys': 120,
            'sbu': 160,
        }
        args.epochs = epoches[args.dataset.lower()]
    if args.lr is None:
        lrs = {
            'coco': 0.004,
            'pascal_aug': 0.001,
            'pascal_voc': 0.0001,
            'pcontext': 0.001,
            'ade20k': 0.01,
            'citys': 0.01,
            'sbu': 0.001,
        }
        args.lr = lrs[args.dataset.lower()] / 8 * args.batch_size
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        # self.device = torch.device(args.device)

        loc = 'npu:{}'.format(args.local_rank)
        if args.device == "npu":
            self.device = torch.npu.set_device(loc)
        else:
            self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
        train_dataset = get_segmentation_dataset(args.dataset, split='train', mode='train', **data_kwargs)
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='val', **data_kwargs)
        args.iters_per_epoch = len(train_dataset) // (args.num_gpus * args.batch_size)
        args.max_iters = args.epochs * args.iters_per_epoch

        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iters)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, args.batch_size)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.BatchNorm2d#nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                            aux=args.aux, jpu=args.jpu, norm_layer=BatchNorm2d).to(loc)

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # create criterion
        self.criterion = get_segmentation_loss(args.model, use_ohem=args.use_ohem, aux=args.aux,
                                               aux_weight=args.aux_weight, ignore_index=-1).to(loc)  # (args.device)

        # optimizer, for model just includes pretrained, head and auxlayer
        params_list = list()
        if hasattr(self.model, 'pretrained'):
            params_list.append({'params': self.model.pretrained.parameters(), 'lr': args.lr})
        if hasattr(self.model, 'exclusive'):
            for module in self.model.exclusive:
                params_list.append({'params': getattr(self.model, module).parameters(), 'lr': args.lr * 10})
        self.optimizer = torch.optim.SGD(params_list,
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        if args.amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=args.opt_level,
                                                        loss_scale=args.loss_scale)

        # lr scheduling
        self.lr_scheduler = WarmupPolyLR(self.optimizer,
                                         max_iters=args.max_iters,
                                         power=0.9,
                                         warmup_factor=args.warmup_factor,
                                         warmup_iters=args.warmup_iters,
                                         warmup_method=args.warmup_method)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                             output_device=args.local_rank, find_unused_parameters=True, broadcast_buffers=False)

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)

        self.best_pred = 0.0

    def train(self):
        batch_time = AverageMeter('Time', ':6.3f', start_count_index=5)

        loc = 'npu:{}'.format(self.args.local_rank)

        save_to_disk = get_rank() == 0
        epochs, max_iters = self.args.epochs, self.args.max_iters
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.args.iters_per_epoch
        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        start_time = time.time()
        logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        end = time.time()

        self.model.train()
        for iteration, (images, targets, _) in enumerate(self.train_loader):
            iteration = iteration + 1
            self.lr_scheduler.step()

            # if 'npu' in CALCULATE_DEVICE:
            targets = targets.to(torch.int32)
            images = images.to(loc)
            targets = targets.to(loc)

            # with torch.autograd.profiler.profile(use_npu=True) as prof:
            outputs = self.model(images)

            loss_dict = self.criterion(outputs, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            self.optimizer.zero_grad()

            # losses.backward()
            if self.args.amp:
                with amp.scale_loss(losses, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                losses.backward()

            self.optimizer.step()

            # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            # prof.export_chrome_trace("output.prof") # "output.prof"为输出文件地址

            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                logger.info(
                "Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                    iteration, max_iters, self.optimizer.param_groups[0]['lr'], losses_reduced.item(),
                    str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))

            if iteration % self.args.iters_per_epoch == 0 and batch_time.avg > 0:
                logger.info("Epoch: {:d}/{:d} || FPS img/s: {:.3f}".format(
                    iteration // self.args.iters_per_epoch, epochs,
                    args.num_gpus * self.args.batch_size / batch_time.avg))

            if iteration % save_per_iters == 0 and save_to_disk and self.args.local_rank == 0:
                save_checkpoint(self.model, self.args, is_best=False)

            batch_time.update(time.time() - end)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation()
                self.model.train()

            end = time.time()

        if self.args.local_rank == 0:
            save_checkpoint(self.model, self.args, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))

    def validation(self):
        loc = 'npu:{}'.format(self.args.local_rank)

        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False
        self.metric.reset()
        #if self.args.distributed:
            #model = self.model.module
        #else:
        model = self.model
        torch.npu.empty_cache()  # TODO check if it helps
        model.eval()
        for i, (image, target, filename) in enumerate(self.val_loader):
            # if 'npu' in CALCULATE_DEVICE:
            #    target = target.to(torch.int32)
            target = target.to(torch.int32)
            image = image.to(loc)
            target = target.to(loc)
            with torch.no_grad():
                outputs = model(image)
            self.metric.update(outputs[0], target)
            pixAcc, mIoU = self.metric.get()
            logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))

        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        if self.args.local_rank == 0:
            save_checkpoint(self.model, self.args, is_best)
        synchronize()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', start_count_index=10):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.start_count_index = start_count_index

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count == 0:
            self.N = n

        self.val = val
        self.count += n
        if self.count > (self.start_count_index * self.N):
            self.sum += val * n
            self.avg = self.sum / (self.count - self.start_count_index * self.N)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = '{}_{}_{}.pth'.format(args.model, args.backbone, args.dataset)
    if args.model == "enet":
        filename = '{}_{}.pth'.format(args.model, args.dataset)
    filename = os.path.join(directory, filename)

    #if args.distributed:
        #model = model.module
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = '{}_{}_{}_best_model.pth'.format(args.model, args.backbone, args.dataset)
        if args.model == "enet":
            best_filename = '{}_{}_best_model.pth'.format(args.model, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()

    os.environ['MASTER_ADDR'] = '127.0.0.1'  # 可以使用当前真实ip或者'127.0.0.1'
    os.environ['MASTER_PORT'] = '29688'  # 随意一个可使用的port即可

    # reference maskrcnn-benchmark
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    #args.num_gpus = 1
    args.distributed = num_gpus > 1

    if args.device == "npu":
        args.device = "npu"
    elif not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        loc = 'npu:{}'.format(args.local_rank)
        torch.npu.set_device(loc)
        torch.distributed.init_process_group(backend=args.dist_backend, init_method="env://")
        synchronize()
    args.lr = args.lr * num_gpus

    logger_filename = '{}_{}_{}_log.txt'.format(args.model, args.backbone, args.dataset)
    if args.model == "enet":
        logger_filename = '{}_{}_log.txt'.format(args.model, args.dataset)
    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(), filename=logger_filename)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    trainer = Trainer(args)
    trainer.train()
    torch.npu.empty_cache()

