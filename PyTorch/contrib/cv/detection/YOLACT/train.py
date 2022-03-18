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
from data import *
from utils.augmentations import SSDAugmentation, BaseTransform
from utils.functions import MovingAverage, SavePath
from utils.logger import Log
from utils import timer
from layers.modules import MultiBoxLoss
from yolact import Yolact
import os
import sys
import time
import math, random
from pathlib import Path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
# Oof
import eval as eval_script
from apex import amp

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Yolact Training Script')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--max_iter', default=800000, type=int,
                    help='max iteration of training')
parser.add_argument('--data', default='/home/data/coco', type=str, 
                    help='the path of training dataset')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from. If this is "interrupt"'\
                         ', the model will resume training from the interrupt file.')
parser.add_argument('--start_iter', default=-1, type=int,
                    help='Resume training at this iter. If this is -1, the iteration will be' \
                         'determined from the file name.')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--node_device', default='npu', type=str, help='choose the calculating device')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning_rate', default=None, type=float,
                    help='Initial learning rate. Leave as None to read this from the config.')
parser.add_argument('--momentum', default=None, type=float,
                    help='Momentum for SGD. Leave as None to read this from the config.')
parser.add_argument('--decay', '--weight_decay', default=None, type=float,
                    help='Weight decay for SGD. Leave as None to read this from the config.')
parser.add_argument('--gamma', default=None, type=float,
                    help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models.')
parser.add_argument('--log_folder', default='logs/',
                    help='Directory for saving logs.')
parser.add_argument('--config', default='yolact_base_config',
                    help='The config object to use.')
parser.add_argument('--save_interval', default=2000, type=int,
                    help='The number of iterations between saving the model.')
parser.add_argument('--validation_size', default=5000, type=int,
                    help='The number of images to use for validation.')
parser.add_argument('--validation_epoch', default=-1, type=int,
                    help='Output validation information every n iterations. If -1, do no validation.')
parser.add_argument('--keep_latest', dest='keep_latest', action='store_true',
                    help='Only keep the latest checkpoint instead of each one.')
parser.add_argument('--keep_latest_interval', default=100000, type=int,
                    help='When --keep_latest is on, don\'t delete the latest file at these intervals. This should be a multiple of save_interval or 0.')
parser.add_argument('--dataset', default=None, type=str,
                    help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
parser.add_argument('--no_log', dest='log', action='store_false',
                    help='Don\'t log per iteration information into log_folder.')
parser.add_argument('--log_gpu', dest='log_gpu', action='store_true',
                    help='Include GPU information in the logs. Nvidia-smi tends to be slow, so set this with caution.')
parser.add_argument('--no_interrupt', dest='interrupt', action='store_false',
                    help='Don\'t save an interrupt when KeyboardInterrupt is caught.')
parser.add_argument('--batch_alloc', default=None, type=str,
                    help='If using multiple GPUS, you can set this to be a comma separated list detailing which GPUs should get what local batch size (It should add up to your total batch size).')
parser.add_argument('--no_autoscale', dest='autoscale', action='store_false',
                    help='YOLACT will automatically scale the lr and the number of iterations depending on the batch size. Set this if you want to disable that.')
parser.add_argument('--train_performance', default=False, type=bool, help='choose running mode, if train_performance is true, this script will not run eval')
parser.add_argument('--useDDP', default=True, type=bool, help='use DistributedDataParallel or not')
parser.add_argument('--seed', default=None, type=int, help='set PyTorch seed')
parser.add_argument('--local_rank', default=0, type=int, help='ranking within the nodes')
parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
args = parser.parse_args()

#设置配置文件，无用
if args.config is not None:
    set_cfg(args.config)

#设置项目数据集，无用
if args.dataset is not None:
    set_dataset(args.dataset)

cfg.max_iter = args.max_iter
args.world_size = int(os.environ['RANK_SIZE'])
#注意这里的batch_size是总batch_size，原来使用DP时，这里是16，会进入该分支
if args.autoscale and (args.batch_size * args.world_size) != 8:
    factor = (args.batch_size * args.world_size) / 8
    if __name__ == '__main__':
        print('Scaling parameters by %.2f to account for a batch size of %d.' % (factor, args.batch_size * args.world_size))

    cfg.lr *= factor
    cfg.max_iter //= factor
    cfg.lr_steps = [x // factor for x in cfg.lr_steps]

# Update training parameters from the config if necessary
def replace(name):
    if getattr(args, name) == None: setattr(args, name, getattr(cfg, name))

#将args中参数替换为config中预设的值，便于后续调用
replace('lr')
replace('decay')
replace('gamma')
replace('momentum')

# This is managed by set_lr
#两个学习率都有用，在后续自动更新学习率中，可以使用
cur_lr = args.lr

#检查环境
if args.node_device == 'npu':
    if torch.npu.device_count() == 0:
        print('No NPUs detected. Exiting...')
        exit(-1)
else:
    if torch.cuda.device_count() == 0:
        print('No GPUs detected. Exiting...')
        exit(-1)

#当一块显卡中的图像个数大于等于6时，才启用batch normalization
if args.node_device == 'npu':
    device_count = torch.npu.device_count()
else:
    device_count = torch.cuda.device_count()
if args.batch_size // device_count < 6 and (not args.useDDP):
    if __name__ == '__main__':
        print('Per-GPU batch size is less than the recommended limit for batch norm. Disabling batch norm.')
    cfg.freeze_bn = True

if args.seed is not None:
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if args.node_device == 'npu':
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print('Finish set seed, seed is :', seed)

loss_types = ['B', 'C', 'M', 'P', 'D', 'E', 'S', 'I']

if args.node_device == 'npu':
    if torch.npu.is_available():
        print("npu environment is okay!, and current device count is", torch.npu.device_count())
else:
    if torch.cuda.is_available():
        print("gpu environment is okay!, and current device count is", torch.cuda.device_count())

class NetLoss(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.

    损失函数模块，YOLACT只使用Multibox Loss，但单独封装NetLoss模块的目的是多卡训练
    """

    def __init__(self, net:Yolact, criterion:MultiBoxLoss):
        super().__init__()

        self.net = net
        self.criterion = criterion

    def forward(self, images, targets, masks, num_crowds):
        preds = self.net(images)
        losses = self.criterion(self.net, preds, targets, masks, num_crowds)
        return losses

class ScatterWrapper:
    """ Input is any number of lists. This will preserve them through a dataparallel scatter. """
    def __init__(self, *args):
        for arg in args:
            if not isinstance(arg, list):
                print('Warning: ScatterWrapper got input of non-list type.')
        self.args = args
        self.batch_size = len(args[0])

    def make_mask(self):
        out = torch.Tensor(list(range(self.batch_size))).long()
        if args.cuda:
            if args.node_device == 'npu':
                return out.npu()
            else:
                return out.cuda()
        else: return out

    def get_args(self, mask):
        device = mask.device
        mask = [int(x) for x in mask]
        out_args = [[] for _ in self.args]

        for out, arg in zip(out_args, self.args):
            for idx in mask:
                x = arg[idx]
                if isinstance(x, torch.Tensor):
                    x = x.to(device)
                out.append(x)

        return out_args

def train(args):
    #创建模型权重文件存储目录
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    args.rank_id = int(os.environ['RANK_ID'])
    args.world_size = int(os.environ['RANK_SIZE'])
    if args.node_device == 'npu':
        args.device = 'npu:' + str(args.rank_id)
        torch.npu.set_device(args.device)
    else:
        args.device = 'cuda:' + str(args.rank_id)
        torch.cuda.set_device(args.device)

    args.is_master_node = args.world_size == 1 or args.rank_id == 0

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29688'
    if args.node_device == 'npu':
        dist.init_process_group(backend='hccl', world_size=args.world_size, rank=args.rank_id)
    else:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank_id)

    cfg.dataset.train_images = os.path.join(args.data, 'train2017')
    cfg.dataset.train_info = os.path.join(args.data, 'annotations/instances_train2017.json')
    cfg.dataset.valid_images = os.path.join(args.data, 'val2017')
    cfg.dataset.valid_info = os.path.join(args.data, 'annotations/instances_val2017.json')
    
    #创建数据集，dataset为训练数据集
    dataset = COCODetection(image_path=cfg.dataset.train_images,
                            info_file=cfg.dataset.train_info,
                            transform=SSDAugmentation(MEANS))

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    if args.validation_epoch > 0:
        #调整eval.py脚本对应参数
        setup_eval()

        #创建数据集，val_dataset为验证数据集，5000张图像
        val_dataset = COCODetection(image_path=cfg.dataset.valid_images,
                                    info_file=cfg.dataset.valid_info,
                                    transform=BaseTransform(MEANS))

    # Parallel wraps the underlying module, but when saving and loading we don't want that
    yolact_net = Yolact()
    net = yolact_net
    net.train()

    if args.log:
        log = Log(cfg.name, args.log_folder, dict(args._get_kwargs()),
                  overwrite=(args.resume is None), log_gpu_stats=args.log_gpu)  #构造日志类

    # I don't use the timer during training (I use a different timing method).
    # Apparently there's a race condition with multiple GPUs, so disable it just to be safe.
    timer.disable_all()

    # Both of these can set args.resume to None, so do them before the check
    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(args.save_folder)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(args.save_folder, cfg.name)

    if args.resume is not None:
        print('Resuming training, loading {}...'.format(args.resume))
        yolact_net.load_weights(args.resume)

        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.resume).iteration
    else:
        print('Initializing weights...')
        yolact_net.init_weights(backbone_path=args.save_folder + cfg.backbone.path)

    #损失函数，multibox loss
    #threshold ： 门限阈值
    #   pos_threshold 即为 ： 高于这个值，那么就说明预测正确足够confident，即可以认为识别正确
    #   pos_threshold 即为： 低于这个值，那么就可以自信认为识别错误

    #ohem_negpos_ratio
    criterion = MultiBoxLoss(num_classes=cfg.num_classes,
                             pos_threshold=cfg.positive_iou_threshold,
                             neg_threshold=cfg.negative_iou_threshold,
                             negpos_ratio=cfg.ohem_negpos_ratio)

    if args.batch_alloc is not None:
        args.batch_alloc = [int(x) for x in args.batch_alloc.split(',')]
        if sum(args.batch_alloc) != args.batch_size:
            print('Error: Batch allocation (%s) does not sum to batch size (%s).' % (args.batch_alloc, args.batch_size))
            exit(-1)

    net = net.to(args.device)
    criterion = criterion.to(args.device)
    #优化器SGD，随机梯度下降法
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    net, optimizer = amp.initialize(net, optimizer, opt_level="O0")
    net = nn.parallel.DistributedDataParallel(net, device_ids=[args.rank_id])

    # Initialize everything
    if not cfg.freeze_bn: yolact_net.freeze_bn() # Freeze bn so we don't kill our means
    if args.node_device == 'npu':
        yolact_net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).npu())
    else:
        yolact_net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda())
    if not cfg.freeze_bn: yolact_net.freeze_bn(True)

    # loss counters
    loc_loss = 0
    conf_loss = 0
    iteration = max(args.start_iter, 0)
    count_fps_iters = 0
    count_fps_time = 0.0
    last_time = time.time()

    epoch_size = len(dataset) // (args.batch_size * args.world_size)
    num_epochs = math.ceil(cfg.max_iter / epoch_size)

    # Which learning rate adjustment step are we on? lr' = lr * gamma ^ step_index
    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False,
                                  collate_fn=detection_collate,
                                  pin_memory=True, sampler=train_sampler)


    save_path = lambda epoch, iteration: SavePath(cfg.name, epoch, iteration).get_path(root=args.save_folder)
    time_avg = MovingAverage()

    global loss_types # Forms the print order
    loss_avgs  = { k: MovingAverage(100) for k in loss_types }

    print('Begin training! RANK_ID :', args.rank_id, '[', time.time(), ']')

    if args.is_master_node:
        print(args)
        print(cfg)
        
    print()
    # try-except so you can use ctrl+c to save early and stop training
    try:
        for epoch in range(num_epochs):
            # Resume from start_iter
            if (epoch+1)*epoch_size < iteration:
                continue

            train_sampler.set_epoch(epoch)

            beginIteration = iteration

            for datum in data_loader:
                # Stop if we've reached an epoch if we're resuming from start_iter
                if iteration == (epoch+1)*epoch_size:
                    break

                # Stop at the configured number of iterations even if mid-epoch
                if iteration == cfg.max_iter:
                    break

                # Change a config setting if we've reached the specified iteration
                changed = False
                for change in cfg.delayed_settings:
                    if iteration >= change[0]:
                        changed = True
                        cfg.replace(change[1])

                        # Reset the loss averages because things might have changed
                        for avg in loss_avgs:
                            avg.reset()

                # If a config setting was changed, remove it from the list so we don't keep checking
                if changed:
                    cfg.delayed_settings = [x for x in cfg.delayed_settings if x[0] > iteration]

                # Warm up by linearly interpolating the learning rate from some smaller value
                if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                    set_lr(optimizer, (args.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)

                # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
                while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                    step_index += 1
                    set_lr(optimizer, args.lr * (args.gamma ** step_index))

                #print('current device:', prep_data_device, 'the batch on this device has id list:', [item[0] for item in datum[0]])
                datum[0] = [item[1] for item in datum[0]]
                images, targets, masks, num_crowds = prepare_data(datum, [args.device])
                out = net(images[0])
                optimizer.zero_grad()
                wrapper = ScatterWrapper(targets, masks, num_crowds)
                losses = criterion(net, out, wrapper, wrapper.make_mask())

                losses = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
                loss = sum([losses[k] for k in losses])

                # no_inf_mean removes some components from the loss, so make sure to backward through all of it
                # all_loss = sum([v.mean() for v in losses.values()])

                # Backprop
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

                if torch.isfinite(loss).item():
                    optimizer.step()
                    #print('\t finish one step! NPU :', args.rank_id, '[', time.time(), ']')

                # Add the loss to the moving average for bookkeeping
                for k in losses:
                    loss_avgs[k].add(losses[k].item())

                cur_time  = time.time()
                elapsed   = cur_time - last_time
                last_time = cur_time

                # Exclude graph setup from the timing information
                if iteration != args.start_iter:
                    time_avg.add(elapsed)

                if iteration % 10 == 0:
                    if count_fps_time == 0.0:
                        avg_fps = 0.0
                    else:
                        avg_fps = (count_fps_iters * args.batch_size * args.world_size) / count_fps_time

                    eta_str = str(datetime.timedelta(seconds=(cfg.max_iter-iteration) * time_avg.get_avg())).split('.')[0]

                    total = sum([loss_avgs[k].get_avg() for k in losses])
                    loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])

                    print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || ETA: %s || timer: %.3f || fps: %.3f || avg_fps: %.3f' + ' || GPU: ' + str(args.rank_id))
                          % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed, (args.batch_size * args.world_size) / elapsed, avg_fps]), flush=True)
                if args.log:
                    precision = 5
                    loss_info = {k: round(losses[k].item(), precision) for k in losses}
                    loss_info['T'] = round(loss.item(), precision)

                    if args.log_gpu:
                        log.log_gpu_stats = (iteration % 10 == 0) # nvidia-smi is sloooow

                    log.log('train', loss=loss_info, epoch=epoch, iter=iteration,
                            lr=round(cur_lr, 10), elapsed=elapsed)

                    log.log_gpu_stats = args.log_gpu

                if epoch == 0:
                    if iteration >= beginIteration + 1000: #第一个epoch前期存在大量动态编译算子，不参与统计
                        count_fps_iters += 1
                        count_fps_time += elapsed

                else:
                    if iteration >= beginIteration + 5: #不统计每个epoch前五个step耗时
                        count_fps_iters += 1
                        count_fps_time += elapsed

                iteration += 1

                if iteration % args.save_interval == 0 and iteration != args.start_iter:
                    if args.keep_latest:
                        latest = SavePath.get_latest(args.save_folder, cfg.name)

                    print('Saving state, iter:', iteration)
                    if args.is_master_node:
                        yolact_net.save_weights(save_path(epoch, iteration))

                    if args.keep_latest and latest is not None:
                        if args.keep_latest_interval <= 0 or iteration % args.keep_latest_interval != args.save_interval:
                            print('Deleting old save...')
                            os.remove(latest)

            # This is done per epoch
            if args.validation_epoch > 0 and args.is_master_node and (not args.train_performance):
                if epoch % args.validation_epoch == 0 and epoch > 0:
                    compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)

        # Compute validation mAP after training is finished
        if args.validation_epoch > 0 and args.is_master_node and (not args.train_performance):
            compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)
    except KeyboardInterrupt:
        if args.interrupt:
            print('Stopping early. Saving network...')

            # Delete previous copy of the interrupted network so we don't spam the weights folder
            SavePath.remove_interrupt(args.save_folder)

            if args.is_master_node:
                yolact_net.save_weights(save_path(epoch, repr(iteration) + '_interrupt'))
        exit()

    if args.is_master_node:
        yolact_net.save_weights(save_path(epoch, iteration))


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    global cur_lr
    cur_lr = new_lr

def gradinator(x):
    x.requires_grad = False
    return x

def prepare_data(datum, devices:list=None, allocation:list=None):
    with torch.no_grad():
        if devices is None:
            devices = ['npu:0'] if args.node_device == 'npu' else ['cuda:0']
        if allocation is None:
            allocation = [args.batch_size // len(devices)] * (len(devices) - 1)
            allocation.append(args.batch_size - sum(allocation)) # The rest might need more/less

        images, (targets, masks, num_crowds) = datum

        cur_idx = 0
        for device, alloc in zip(devices, allocation):
            for _ in range(alloc):
                images[cur_idx]  = gradinator(images[cur_idx].to(device))
                targets[cur_idx] = gradinator(targets[cur_idx].to(device))
                masks[cur_idx]   = gradinator(masks[cur_idx].to(device))
                cur_idx += 1

        if cfg.preserve_aspect_ratio:
            # Choose a random size from the batch
            _, h, w = images[random.randint(0, len(images)-1)].size()

            for idx, (image, target, mask, num_crowd) in enumerate(zip(images, targets, masks, num_crowds)):
                images[idx], targets[idx], masks[idx], num_crowds[idx] \
                    = enforce_size(image, target, mask, num_crowd, w, h)

        cur_idx = 0
        split_images, split_targets, split_masks, split_numcrowds \
            = [[None for alloc in allocation] for _ in range(4)]

        for device_idx, alloc in enumerate(allocation):
            split_images[device_idx]    = torch.stack(images[cur_idx:cur_idx+alloc], dim=0)
            split_targets[device_idx]   = targets[cur_idx:cur_idx+alloc]
            split_masks[device_idx]     = masks[cur_idx:cur_idx+alloc]
            split_numcrowds[device_idx] = num_crowds[cur_idx:cur_idx+alloc]

            cur_idx += alloc

        return split_images, split_targets, split_masks, split_numcrowds

def no_inf_mean(x:torch.Tensor):
    """
    Computes the mean of a vector, throwing out all inf values.
    If there are no non-inf values, this will return inf (i.e., just the normal mean).
    """

    no_inf = [a for a in x if torch.isfinite(a)]

    if len(no_inf) > 0:
        return sum(no_inf) / len(no_inf)
    else:
        return x.mean()

def compute_validation_loss(net, data_loader, criterion):
    global loss_types

    with torch.no_grad():
        losses = {}

        # Don't switch to eval mode because we want to get losses
        iterations = 0
        for datum in data_loader:
            images, targets, masks, num_crowds = prepare_data(datum)
            out = net(images)

            wrapper = ScatterWrapper(targets, masks, num_crowds)
            _losses = criterion(out, wrapper, wrapper.make_mask())

            for k, v in _losses.items():
                v = v.mean().item()
                if k in losses:
                    losses[k] += v
                else:
                    losses[k] = v

            iterations += 1
            if args.validation_size <= iterations * args.batch_size:
                break

        for k in losses:
            losses[k] /= iterations


        loss_labels = sum([[k, losses[k]] for k in loss_types if k in losses], [])
        print(('Validation ||' + (' %s: %.3f |' * len(losses)) + ')') % tuple(loss_labels), flush=True)

def compute_validation_map(epoch, iteration, yolact_net, dataset, log:Log=None):
    with torch.no_grad():
        yolact_net.eval()

        start = time.time()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        val_info = eval_script.evaluate(yolact_net, dataset, train_mode=True, device=args.node_device)
        end = time.time()

        if log is not None:
            log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)

        yolact_net.train()

def setup_eval():
    eval_script.parse_args(['--no_bar', '--max_images='+str(args.validation_size)])

if __name__ == '__main__':
    train(args)
