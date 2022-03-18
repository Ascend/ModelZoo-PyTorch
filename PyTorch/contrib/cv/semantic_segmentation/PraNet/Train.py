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
import torch
from torch.autograd import Variable
import os
import random
import time
import argparse
import warnings
from datetime import datetime
from lib.PraNet_Res2Net import PraNet
from utils.dataloader import get_loader
from utils.utils import AverageMeter, clip_gradient, adjust_lr, AvgMeter, Conv2dForAvgPool2d
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
import apex
from apex import amp
from apex.optimizers import NpuFusedAdam

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int,
                    default=20, help='epoch number')
parser.add_argument('--lr', type=float,
                    default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int,
                    default=16, help='training batch size')
parser.add_argument('--trainsize', type=int,
                    default=352, help='training dataset size')
parser.add_argument('--clip', type=float,
                    default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float,
                    default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int,
                    default=50, help='every n epochs decay learning rate')
parser.add_argument('--train_path', type=str,
                    default='./data/TrainDataset', help='path to train dataset')
parser.add_argument('--train_save', type=str,
                    default='PraNet_Res2Net')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default=False, dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--device', default='gpu', type=str, help='npu or gpu')
parser.add_argument('--addr', default='127.0.0.1',
                    type=str, help='master addr')
parser.add_argument('--device_list', default='0,1,2,3,4,5,6,7',
                    type=str, help='device id list')
parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--loss-scale', default=128.0, #type=float,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--opt-level', default='O2', type=str,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--prof', default=False, action='store_true',
                help='use profiling to evaluate the performance of model')
args = parser.parse_args()

def structure_loss(pred, mask, args):
    # if args.device == "npu":
    #     mask = mask.to("cpu")
    # weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    avg = Conv2dForAvgPool2d(mask.shape[1], kernel_size=31, stride=1, padding=15)
    weit = 1 + 5*torch.abs(avg(mask) - mask)
    # if args.device == "npu":
    #     loc = 'npu:{}'.format(args.gpu)
    #     mask = mask.to(loc)
    #     weit = weit.to(loc)
    
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def train(train_loader, model, optimizer, epoch, ngpus_per_node, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    end = time.time()
    model.train()
    total_step = len(train_loader)

    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            # images = Variable(images).cuda()
            # gts = Variable(gts).cuda()
            if args.device == 'npu':
                loc = 'npu:{}'.format(args.gpu)
                images = images.to(loc, non_blocking=True)
                gts = gts.to(loc, non_blocking=True)
            else:
                images = images.cuda(args.gpu, non_blocking=True)
                gts = gts.cuda(args.gpu, non_blocking=True)
            # ---- rescale ----
            trainsize = int(round(args.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
            # ---- loss function ----
            loss5 = structure_loss(lateral_map_5, gts, args)
            loss4 = structure_loss(lateral_map_4, gts, args)
            loss3 = structure_loss(lateral_map_3, gts, args)
            loss2 = structure_loss(lateral_map_2, gts, args)
            loss = loss2 + loss3 + loss4 + loss5    # TODO: try different weights for loss
            # ---- backward ----
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            clip_gradient(optimizer, args.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record2.update(loss2.data, args.batchsize)
                loss_record3.update(loss3.data, args.batchsize)
                loss_record4.update(loss4.data, args.batchsize)
                loss_record5.update(loss5.data, args.batchsize)
        # measure elapsed time
        cost_time = time.time() - end
        batch_time.update(cost_time)
        end = time.time()
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, args.epoch, i, total_step,
                         loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))
    save_path = 'snapshots/{}/'.format(args.train_save)
    os.makedirs(save_path, exist_ok=True)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.gpu % ngpus_per_node == 0):
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), save_path + 'PraNet-%d.pth' % epoch)
            print('[Saving Snapshot:]', save_path + 'PraNet-%d.pth'% epoch)
        if batch_time.avg:
            # print("[npu id:", args.gpu, "]", "batch_size:", args.world_size * args.batch_size,
            #         'Time: {:.3f}'.format(batch_time.avg), '* FPS@all {:.3f}'.format(
            #         args.batch_size * args.world_size / batch_time.avg))
            print('Time: {:.3f}'.format(batch_time.avg), '* FPS@all {:.3f}'.format(
                    args.batchsize * args.world_size / batch_time.avg))

def profiling(data_loader, model, optimizer, args):
    # switch to train mode
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]

    def update(model, images, gts, optimizer):
        for rate in size_rates:
            optimizer.zero_grad()
                # ---- rescale ----
            trainsize = int(round(args.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
            # ---- loss function ----
            loss5 = structure_loss(lateral_map_5, gts, args)
            loss4 = structure_loss(lateral_map_4, gts, args)
            loss3 = structure_loss(lateral_map_3, gts, args)
            loss2 = structure_loss(lateral_map_2, gts, args)
            loss = loss2 + loss3 + loss4 + loss5    # TODO: try different weights for loss
            # ---- backward ----
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            # optimizer.zero_grad()
            clip_gradient(optimizer, args.clip)
            optimizer.step()
        print("#"*20, "train end ", "#"*20, step)

    for step, pack in enumerate(data_loader):
        
            images, gts = pack

            if args.device == 'npu':
                loc = 'npu:{}'.format(args.gpu)
                images = images.to(loc, non_blocking=True)
                gts = gts.to(loc, non_blocking=True)
            else:
                # images = images.cuda(args.gpu, non_blocking=True)
                # gts = target.cuda(args.gpu, non_blocking=True)
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()

            if step < 5:
                update(model, images, gts, optimizer)
            else:
                if args.device == 'npu':
                    with torch.autograd.profiler.profile(use_npu=True) as prof:
                        update(model, images, gts, optimizer)
                        print("#"*20, "end profiling", "#"*20, step)
                else:
                    with torch.autograd.profiler.profile(use_cuda=True) as prof:
                        update(model, images, gts, optimizer)
                        print("#"*20, "end profiling", "#"*20, step)
                break

    prof.export_chrome_trace("output.prof")

def main():
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '28688'

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                    'disable data parallelism.')

    # if args.dist_url == "env://" and args.world_size == -1:
    #     args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.device == 'npu':
        if args.distributed:
            ngpus_per_node = 8
        else:
            ngpus_per_node = 1
    else:
        ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node:', ngpus_per_node)

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use N/GPU: {} for training".format(args.gpu))
    
    if args.distributed:
        # if args.dist_url == "env://" and args.rank == -1:
        #     args.rank = int(os.environ["RANK"])
        # print("distributed, torch.distributed.init_process_group")
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        if args.device == 'npu':
            dist.init_process_group(backend=args.dist_backend,  # init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)
        else:
            dist.init_process_group(backend=args.dist_backend, init_method="env://",
                                    world_size=args.world_size, rank=args.rank)

    if args.pretrained:
        print("=> using pre-trained model", '#'*20)
        model = PraNet()
        pretrained_dict = torch.load("./snapshots/PraNet_Res2Net/PraNet-19.pth", map_location="cpu")
        model.load_state_dict({k.replace('module.',''):v for k, v in pretrained_dict.items()})
        if "fc.weight" in pretrained_dict:
            pretrained_dict.pop('fc.weight')
            pretrained_dict.pop('fc.bias')
        model.load_state_dict(pretrained_dict, strict=False)
    else:
        model = PraNet()

    # model to device 
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            if args.device == 'npu':
                loc = 'npu:{}'.format(args.gpu)
                torch.npu.set_device(loc)
                model = model.to(loc)
            else:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
            args.batchsize = int(args.batchsize / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        else:
            if args.device == 'npu':
                loc = 'npu:{}'.format(0)
                model = model.to(loc)
            else:
                model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            print("[gpu id:", args.gpu, "]",
                  "============================test   args.gpu is not None   else==========================")
    else:
        if args.device == 'npu':
            loc = 'npu:{}'.format(args.gpu)
            torch.npu.set_device(loc)
            model = model.to(args.gpu)
        else:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)

    params = model.parameters()
    # optimizer = torch.optim.Adam(params, args.lr)
    
    optimizer = NpuFusedAdam(
        model.parameters(),
        lr=args.lr
    )

    if args.amp:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale)

    # model ddp
    if args.distributed:
        if args.gpu is not None:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False,
                                                                  find_unused_parameters=True)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                if args.device == 'npu':
                    loc = 'npu:{}'.format(args.gpu)
                else:
                    loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            # best_acc1 = checkpoint['best_acc1']
            # if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                # best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.amp:
                amp.load_state_dict(checkpoint['amp'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # DateLoader
    image_root = '{}/images/'.format(args.train_path)
    gt_root = '{}/masks/'.format(args.train_path)
    train_sampler, train_loader = get_loader(image_root, gt_root, batchsize=args.batchsize, trainsize=args.trainsize, distributed=args.distributed)
    total_step = len(train_loader)

    if args.prof:
        print("#"*20, "start profiling", "#"*20)
        profiling(train_loader, model, optimizer, args)
        return 

    start_time = time.time()
    if args.gpu == 0:
        print("#"*20, "Start Training", "#"*20, total_step, "args.gnpu = ", args.gpu)

    for epoch in range(1, args.epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_lr(optimizer, args.lr, epoch, args.decay_rate, args.decay_epoch)
        train(train_loader, model, optimizer, epoch, ngpus_per_node, args)

        # save_checkpoint 
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            if args.amp:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': 'wide_resnet50_2',
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict(),
                })
            else:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': 'wide_resnet50_2',
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                })

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

if __name__ == '__main__':
    main()