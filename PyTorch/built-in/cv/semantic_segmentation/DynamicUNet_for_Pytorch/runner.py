# Copyright 2022 Huawei Technologies Co., Ltd
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

import argparse
import time
import datetime
import os
import shutil
import sys
import ast

import torch
if torch.__version__ >= "1.8":
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
import apex
import bugfix

import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
try:
    from torch_npu.utils.profiler import Profile
except ImportError:
    print("Profile not in torch_npu.utils.profiler now... Auto Profile disabled.", flush=True)
    class Profile:
        def __init__(self, *paramers, **kwargs):
            pass
        def start(self):
            pass
        def end(self):
            pass
from torchvision import transforms
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.loss import get_segmentation_loss
from core.utils.distributed import *
from core.utils.logger import setup_logger
from core.utils.lr_scheduler import WarmupPolyLR
from core.utils.score import SegmentationMetric
from dynamic_unet import get_dynamicunet

# set seed
import numpy as np
import random
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class NoProfiling(object):
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='fcn',
                        choices=['fcn32s', 'fcn16s', 'fcn8s',
                                 'fcn', 'psp', 'deeplabv3', 'deeplabv3_plus',
                                 'danet', 'denseaspp', 'bisenet',
                                 'encnet', 'dunet', 'icnet',
                                 'enet', 'ocnet', 'ccnet', 'psanet',
                                 'cgnet', 'espnet', 'lednet', 'dfanet', 'dynamicunet'],
                        help='model name (default: fcn32s)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['vgg16', 'resnet18', 'resnet50',
                                 'resnet101', 'resnet152', 'densenet121',
                                 'densenet161', 'densenet169', 'densenet201'],
                        help='backbone name (default: vgg16)')
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        choices=['pascal_voc', 'pascal_aug', 'ade20k',
                                 'citys', 'sbu'],
                        help='dataset name (default: pascal_voc)')
    parser.add_argument('--dataset-path', type=str, default='./',
                        help='dataset path')
    parser.add_argument('--pretrained', type=str, default='',
                        help='pretrained model path')
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

    # Apex
    parser.add_argument('--amp', action='store_true', default=False,
                        help='enable auto mixed precision')

    # perf
    parser.add_argument('--perf-only', action='store_true', default=False,
                        help='enable perf mode')
    parser.add_argument('--perf_iter', type=int, default=-1,
                        help='if >0 set --perf-only=True and iteration number')
    # bin model
    parser.add_argument('--bin_mode', default="True", type=ast.literal_eval,
                        help='enable bin model')
    # profiling
    parser.add_argument('--profiling', type=str, default="None",
                        help='choose profiling way--CANN,GE,None')

    parser.add_argument('--start_step', type=int, default=0,
                        help='profiling start step')
    parser.add_argument('--stop_step', type=int, default=100,
                        help='profiling stop_ step')
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
        self.device = torch.device(args.device)
        print("Device Info: ",self.device)
        self.amp = args.amp

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader
        data_kwargs = {'root': args.dataset_path, 'transform': input_transform, 'base_size': args.base_size,
                       'crop_size': args.crop_size}
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
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d

        if args.model == 'dynamicunet':
            self.model = get_dynamicunet(args=args, dataset=args.dataset).to(self.device)
        else:
            self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                                aux=args.aux, jpu=args.jpu, norm_layer=BatchNorm2d).to(self.device)

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # create criterion
        self.criterion = get_segmentation_loss(args.model, use_ohem=args.use_ohem, aux=args.aux,
                                               aux_weight=args.aux_weight, ignore_index=-1).to(self.device)

        # optimizer, for model just includes pretrained, head and auxlayer
        params_list = list()
        if args.model == 'dynamicunet':
            pretrained_parameters = []
            head_parameters = []
            for name, parameter in self.model.named_parameters():
                if 'layer.0.' in name:
                    pretrained_parameters.append(parameter)
                else:
                    head_parameters.append(parameter)
            params_list.append({'params': pretrained_parameters, 'lr': args.lr})
            params_list.append({'params': head_parameters, 'lr': args.lr * 10.})
        else:
            if hasattr(self.model, 'pretrained'):
                params_list.append({'params': self.model.pretrained.parameters(), 'lr': args.lr})
            if hasattr(self.model, 'exclusive'):
                for module in self.model.exclusive:
                    params_list.append({'params': getattr(self.model, module).parameters(), 'lr': args.lr * 10})

        if self.amp:
            self.optimizer = apex.optimizers.NpuFusedSGD(params_list,
                                                         lr=args.lr,
                                                         momentum=args.momentum,
                                                         weight_decay=args.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(params_list,
                                             lr=args.lr,
                                             momentum=args.momentum,
                                             weight_decay=args.weight_decay)

        if self.amp:
            self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer, opt_level='O1',
                                                             loss_scale=None,
                                                             combine_grad=True)

        # lr scheduling
        self.lr_scheduler = WarmupPolyLR(self.optimizer,
                                         max_iters=args.max_iters,
                                         power=0.9,
                                         warmup_factor=args.warmup_factor,
                                         warmup_iters=args.warmup_iters,
                                         warmup_method=args.warmup_method)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                             output_device=args.local_rank)

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)

        self.best_pred = 0.0

    def train(self):
        save_to_disk = get_rank() == 0
        epochs, max_iters = self.args.epochs, self.args.max_iters
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.args.iters_per_epoch
        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        end_step = min(self.args.iters_per_epoch if self.args.perf_only else -1, 50)
        # npu 调试参数：导出debug日志时，设置较小迭代次数，防止日志过大
        if self.args.perf_iter > 0:
            end_step=self.args.perf_iter

        start_time = time.time()
        iter_end_time = start_time
        iter_time = 0
        logger.info(
            'Start training, Total Epochs: {:d} = Total Iterations {:d}, \
                Iter Per Epoch: {:d}'.format(epochs, max_iters, self.args.iters_per_epoch))

        self.model.train()
        profiler = Profile(start_step=int(os.getenv("PROFILE_START_STEP", 10)),
                           profile_type=os.getenv("PROFILE_TYPE"))
        for iteration, (images, targets, _) in enumerate(self.train_loader):
            iteration = iteration + 1
            if iteration > self.args.stop_step and self.args.profiling != "None":
                sys.exit()

            self.lr_scheduler.step()
            images = images.to(self.device)
            targets = targets.to(self.device)
            profiler.start()
            outputs = self.model(images)
            loss_dict = self.criterion(outputs, targets)
            losses = sum(loss for loss in loss_dict.values())
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            self.optimizer.zero_grad()
            if self.amp:
                with apex.amp.scale_loss(losses, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                losses.backward()
            self.optimizer.step()
            profiler.end()
            torch.cuda.synchronize()

            current_iter_time = time.time() - iter_end_time
            if iteration < 5:
                iter_time = current_iter_time
            elif iteration % val_per_iters in [0, 1, 2]:
                iter_time = iter_time
            else:
                iter_time = iter_time * 0.9 + current_iter_time * 0.1
            fps = self.args.batch_size * self.args.num_gpus / iter_time

            iter_end_time = time.time()
            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                logger.info(
                    "Iters: {:d}/{:d} || FPS: {:.2f} || Lr: {:.6f} || Loss: {:.4f} || Cost Time: {} || Estimated Time: {}|| Iter_Time: {}".format(
                        iteration, max_iters, fps, self.optimizer.param_groups[0]['lr'], losses_reduced.item(),
                        str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string, iter_time))

            if iteration == end_step:
                exit()

            if iteration % save_per_iters == 0 and save_to_disk:
                save_checkpoint(self.model, self.args, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation()
                self.model.train()

        save_checkpoint(self.model, self.args, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))

    def validation(self):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False
        self.metric.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)
            self.metric.update(outputs[0], target)
            pixAcc, mIoU = self.metric.get()
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))

        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        save_checkpoint(self.model, self.args, is_best)
        synchronize()


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}_{}.pth'.format(args.model, args.backbone, args.dataset)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = '{}_{}_{}_best_model.pth'.format(args.model, args.backbone, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()
    if args.bin_mode:
        torch.npu.set_compile_mode(jit_compile=False)
        print("enble bin mode")
    else:
        print("disable bin mode")

    # reference maskrcnn-benchmark
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        NPU_CALCULATE_DEVICE = os.getenv('ASCEND_DEVICE_ID', 0)
        args.device = "cuda:{}".format(NPU_CALCULATE_DEVICE)
        torch.cuda.set_device(args.device)
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    args.lr = args.lr * num_gpus

    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(), filename='{}_{}_{}_log.txt'.format(
        args.model, args.backbone, args.dataset))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
