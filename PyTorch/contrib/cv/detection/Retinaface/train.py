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
from __future__ import print_function
import os
import torch
if torch.__version__ >= '1.8':
    import torch_npu
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.multiprocessing as mp

from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time

import shutil
import numpy as np

from test_widerface import remove_prefix
from models.retinaface import RetinaFace
from multi_epochs_dataloader import MultiEpochsDataLoader
from apex import amp

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--data', default='train/label.txt',
                    help='Training dataset directory')
parser.add_argument('--val-data', default='val/label.txt',
                    help='val dataset directory')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume-net', default=None, help='resume net for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
parser.add_argument('--dist_backend', default='nccl', type=str, help='GPU for nccl, NPU for hccl')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:50000', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--gpu', default=None, type=int, help='NPU id to use.')
parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
parser.add_argument('--addr', default='127.0.0.1', type=str, help='master addr')
parser.add_argument('--device_num', default=-1, type=int, help='device_num')
parser.add_argument('--device-list', default='0', type=str, help='device id list')

parser.add_argument('--amp', default=False, action='store_true', help='use amp to train the model')
parser.add_argument('--loss-scale', default=64., type=float, help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--opt-level', default='O2', type=str, help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--warmup_epoch', default=1, type=int, help='warm up')
parser.add_argument('--hf32', default=False, action='store_true', help='enable_hi_float_32_execution')
parser.add_argument('--fp32', default=False, action='store_true', help='disable_hi_float_32_execution')
parser.add_argument('--distributed', action='store_true', help='distributed')
parser.add_argument('--max_steps', default=None, type=int, metavar='N',
                        help="number of total steps to run")
args = parser.parse_args()
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50

rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
max_epoch = args.epochs
gpu_train = cfg['gpu_train']

momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
save_folder = args.save_folder

distributed = args.distributed


def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id
    return process_device_map


print(args.device_list)
process_device_map = device_id_to_process_device_map(args.device_list)


def main():
    if args.fp32:
        torch.npu.config.allow_internal_format = False
        torch.npu.conv.allow_hf32 = False
        torch.npu.matmul.allow_hf32 = False
    if args.hf32:
        torch.npu.config.allow_internal_format = False
    option = {}
    if args.fp32 == False and args.hf32 == False:
        option["ACL_PRECISION_MODE"] = "allow_fp32_to_fp16"
    torch.npu.set_option(option)
    print("===============main()=================")
    print(args)
    print("===============main()=================")

    os.environ['LOCAL_DEVICE_ID'] = str(0)
    print("+++++++++++++++++++++++++++LOCAL_DEVICE_ID:", os.environ['LOCAL_DEVICE_ID'])

    os.environ['MASTER_ADDR'] = args.addr  # '10.136.181.51'
    os.environ['MASTER_PORT'] = '29688'

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if args.device_list != '':
        ngpus_per_node = len(args.device_list.split(','))
    elif args.device_num != -1:
        ngpus_per_node = args.device_num
    elif args.device == 'npu':
        ngpus_per_node = torch.npu.device_count()
    else:
        ngpus_per_node = torch.cuda.device_count()
    if args.distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        # The child process uses the environment variables of the parent process,
        # we have to set LOCAL_DEVICE_ID for every proc
        if args.device == 'npu':
            # main_worker(args.gpu, ngpus_per_node, args)
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        else:
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global min_loss
    min_loss = 1e+8
    if args.device_list != '' and args.distributed:
        args.gpu = process_device_map[gpu]
    else:
        args.gpu = int(args.device_list)
    if args.opt_level == 'O0':
        torch.npu.config.allow_internal_format=False
    print("[npu id:", args.gpu, "]", "++++++++++++++++ before set LOCAL_DEVICE_ID:", os.environ['LOCAL_DEVICE_ID'])
    os.environ['LOCAL_DEVICE_ID'] = str(args.gpu)
    print("[npu id:", args.gpu, "]", "++++++++++++++++ LOCAL_DEVICE_ID:", os.environ['LOCAL_DEVICE_ID'])

    if args.gpu is not None:
        print("[gpu id:", args.gpu, "]", "Use GPU: {} for training".format(args.gpu))
    print("----distributed {} ----".format(str(args.distributed)))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        args.rank = args.rank * ngpus_per_node + gpu
        if args.device == 'npu':
            print("init process")
            torch.distributed.init_process_group(backend=args.dist_backend,
                                    world_size=args.world_size, rank=args.rank)
        else:
            torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)

    if args.device =='npu':
        loc = 'npu:{}'.format(args.gpu)
    elif args.device =='gpu':
        loc = 'cuda:{}'.format(args.gpu)
    else:
        loc = 'cpu'

    print("----loc is {}----".format(loc))
    torch.npu.set_device(loc)

    args.batch_size = int(args.batch_size / args.world_size)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    print("[npu id:", args.gpu, "]", "===============main_worker()=================")
    print("[npu id:", args.gpu, "]", args)
    print("[npu id:", args.gpu, "]", "===============main_worker()=================")

    train_loader, train_loader_len, train_sampler = get_pytorch_loader(args.data,
                                                                             args.batch_size,
                                                                             workers=args.workers,
                                                                             distributed=args.distributed)

    net = RetinaFace(cfg=cfg)
    print("Printing net...")
    print(net)
    net = net.to(loc)

    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward()
    priors = priors.to(loc)
    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
    criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)
    if args.amp:
        print("---use amp---")
        net, optimizer = amp.initialize(net, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale)

    if args.distributed:
        print("---use distributed")
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], broadcast_buffers=False)
    start_epoch = 0
    if args.resume_net is not None:
        print('Loading resume network...')
        state_dict = torch.load(args.resume_net, map_location=loc)
        # create new OrderedDict that does not contain `module.`
        if 'state_dict' in state_dict.keys() and not args.distributed:
            new_state_dict = remove_prefix(state_dict['state_dict'], 'module.')
        else:
            new_state_dict = state_dict['state_dict']
        net.load_state_dict(new_state_dict)
        start_epoch = int(state_dict['epoch']) + 1
        min_loss = state_dict['min_loss']
        if args.amp:
            amp.load_state_dict(state_dict['amp'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume_net, start_epoch))
    # optionally resume from a checkpoint
    cudnn.benchmark = True
    step_index = 0
    for epoch in range(start_epoch, max_epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        loss_train = train(train_loader, priors, step_index, train_loader_len, net, criterion,
                           optimizer, epoch, args, ngpus_per_node)
        if ((epoch+1) % 10 == 0 and (epoch+1) > 0) or ((epoch+1) % 5 == 0 and (epoch+1) > cfg['decay1']):
            is_best = loss_train < min_loss
            min_loss = min(loss_train, loss_train)
            if not args.distributed or (args.distributed and args.gpu == 0):
                if args.amp:
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': net.state_dict(),
                        'min_loss': min_loss,
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict(),
                    }, is_best,
                        save_folder + cfg['name'] + '_epoch_' + str(epoch+1) + '_distributed_' + str(distributed)
                        + '.pth.tar')
                else:
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': net.state_dict(),
                        'min_loss': min_loss,
                        'optimizer': optimizer.state_dict(),
                    }, is_best,
                        save_folder + cfg['name'] + '_epoch_' + str(epoch+1) + '_distributed_' + str(distributed)
                        + '.pth.tar')



def get_pytorch_loader(dataset, batch_size, workers=5, _worker_init_fn=None, distributed=False):
    train_dataset = WiderFaceDetection(dataset, preproc(img_dim, rgb_mean))
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    dataloader_fn = MultiEpochsDataLoader
    train_loader = dataloader_fn(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, worker_init_fn=_worker_init_fn, pin_memory=False,
        sampler=train_sampler, collate_fn=detection_collate, drop_last=True)
    return train_loader, len(train_loader), train_sampler


def detection_collate(batch):
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)
    return torch.stack(imgs, 0), targets


def train(train_loader, priors, step_index, train_loader_len, model, criterion, optimizer, epoch, args, ngpus_per_node):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e', start_count_index=0)
    progress = ProgressMeter(
        train_loader_len,
        [batch_time, data_time, losses],
        prefix="Epoch: [{}] MaxEpoch: [{}]".format(epoch + 1, max_epoch))

    if args.device == 'npu':
        loc = 'npu:{}'.format(args.gpu)
    elif args.device == 'gpu':
        loc = 'cuda:{}'.format(args.gpu)
    else:
        loc = 'cpu'
    # switch to train mode
    model.train()
    end = time.time()
    optimizer.zero_grad()

    print('==========step per epoch======================', train_loader_len)
    stepvalues = (cfg['decay1'], cfg['decay2'])

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if epoch in stepvalues:
            step_index += 1
        adjust_learning_rate(optimizer, gamma, epoch, step_index, train_loader_len* epoch + i, train_loader_len)

        images = images.to(loc, non_blocking=True)
        targets = [anno.to(loc) for anno in target]
        optimizer.zero_grad()
        # compute output
        if i == 11 and not args.distributed:
            print("prof out file")
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                out = model(images)
                # backprop
                loss_l, loss_c, loss_landm = criterion(out, priors, targets, args.gpu)
                loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
                if args.amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
            prof.export_chrome_trace("output.prof")
        else:
            out = model(images)
            loss_l, loss_c, loss_landm = criterion(out, priors, targets, args.gpu)
            loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

        losses.update(loss.item(), images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        # measure elapsed time
        if not args.distributed or (args.distributed and args.gpu == 0):
            progress.display(i)
        if args.max_steps and i >= args.max_steps:
            break
    if not args.distributed or (args.distributed and args.gpu == 0):
        if batch_time.avg > 0:
            print("[npu id:", args.gpu, "]",
                  '* FPS@all {:.3f} TIME@all {:.3f}'.format(ngpus_per_node * args.batch_size / batch_time.avg,
                                                             batch_time.avg))
    return losses.avg


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch <= args.warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * args.warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_min_loss%.4f_epoch%d.pth.tar' % (state['min_loss'], state['epoch']))


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("[npu id:", os.environ['LOCAL_DEVICE_ID'], "]", '\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


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




if __name__ == '__main__':
    main()
