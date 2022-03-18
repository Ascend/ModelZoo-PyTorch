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

from __future__ import print_function
import os
import torch
import torch.optim as optim
import argparse
import torch.utils.data as data
from data import AnnotationTransform, VOCDetection, detection_collate, preproc, cfg
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import math
from models.faceboxes import FaceBoxes
from apex import amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import warnings
from multi_epochs_dataloader import MultiEpochsDataLoader

parser = argparse.ArgumentParser(description='FaceBoxes Training')
parser.add_argument('--training_dataset',
                    default='./data/WIDER_FACE', help='Training dataset directory')
parser.add_argument('--data', metavar='DIR', default='./data/PASCAL',
                    help='path to dataset')
parser.add_argument('-b', '--batch_size', default=32,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--ngpu', default=2, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate', default=1e-3,
                    type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None,
                    help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int,
                    help='resume iter for retraining')
parser.add_argument('-max', '--max_epoch', default=300,
                    type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/',
                    help='Location to save checkpoint models')
parser.add_argument('--amp', default=True, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456',
                    type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl',
                    type=str, help='distributed backend')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--device_num', default=-1, type=int, help='device_num')
parser.add_argument('-ef', '--eval-freq', default=5, type=int,
                    metavar='N', help='evaluate frequency (default: 5)')
parser.add_argument('--device_list', default='',
                    type=str, help='device id list')
parser.add_argument('--addr', default='127.0.0.1',
                    type=str, help='master addr')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--opt_level', default='O2', type=str,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--loss_scale', default=128, type=float,
                    help='loss scale using in amp, default -1 means dynamic')
args = parser.parse_args()
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

img_dim = 1024  # only 1024 is supported
rgb_mean = (104, 117, 123)  # bgr order
num_classes = 2
num_gpu = args.ngpu
num_workers = args.num_workers
batch_size = args.batch_size
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
max_epoch = args.max_epoch
training_dataset = args.training_dataset
save_folder = args.save_folder
gpu_train = cfg['gpu_train']


def main():
    args = parser.parse_args()
    print("===============main()=================")
    print(args)
    print("===============main()=================")
    os.environ['LOCAL_DEVICE_ID'] = str(0)
    print("+++++++++++++++++++++++++++LOCAL_DEVICE_ID:",
          os.environ['LOCAL_DEVICE_ID'])

    os.environ['MASTER_ADDR'] = args.addr  # '10.136.181.51'
    os.environ['MASTER_PORT'] = '29688'

    if args.gpu is not None:
        warnings.warn(
            'You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.device_list != '':
        ngpus_per_node = len(args.device_list.split(','))
    elif args.device_num != -1:
        ngpus_per_node = args.device_num
    elif args.device == 'npu':
        ngpus_per_node = torch.npu.device_count()
    else:
        ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        # The child process uses the environment variables of the parent process,
        # we have to set LOCAL_DEVICE_ID for every proc
        if args.device == 'npu':
            # main_worker(args.gpu, ngpus_per_node, args)
            mp.spawn(main_worker, nprocs=ngpus_per_node,
                     args=(ngpus_per_node, args))
        else:
            mp.spawn(main_worker, nprocs=ngpus_per_node,
                     args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    if args.device_list != '':
        args.gpu = int(args.device_list.split(',')[gpu])
    else:
        args.gpu = gpu

    print("[npu id:", args.gpu, "]",
          "++++++++++++++++ before set LOCAL_DEVICE_ID:", os.environ['LOCAL_DEVICE_ID'])
    os.environ['LOCAL_DEVICE_ID'] = str(args.gpu)
    print("[npu id:", args.gpu, "]",
          "++++++++++++++++ LOCAL_DEVICE_ID:", os.environ['LOCAL_DEVICE_ID'])

    if args.gpu is not None:
        print("[npu id:", args.gpu, "]",
              "Use NPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu

        if args.device == 'npu':
            dist.init_process_group(
                backend=args.dist_backend, world_size=args.world_size, rank=args.rank)
        else:
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)

    loc = 'npu:{}'.format(args.gpu)
    torch.npu.set_device(loc)

    args.batch_size = int(args.batch_size / args.world_size)
    args.num_workers = int(
        (args.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    print("[npu id:", args.gpu, "]",
          "===============main_worker()=================")
    print("[npu id:", args.gpu, "]", args)
    print("[npu id:", args.gpu, "]",
          "===============main_worker()=================")

    train_loader, train_loader_len, train_sampler = get_pytorch_train_loader(training_dataset, args.batch_size, num_workers,
                                                                             distributed=args.distributed)

    # creating model
    net = FaceBoxes('train', img_dim, num_classes)
    net = net.to(loc)
    print("Printing net...")
    print(net)
    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.to(loc)
    criterion = MultiBoxLoss(num_classes, 0.35, True,
                             0, True, 7, 0.35, False).to(loc)
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    if args.amp:
        net, optimizer = amp.initialize(
            net, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale)

    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[args.gpu], broadcast_buffers=False)

    start_epoch = 0
    if args.resume_net is not None:
        print('Loading resume network...')
        state_dict = torch.load(args.resume_net, map_location=loc)
        if not args.distributed and 'state_dict' in state_dict.keys():
            new_state_dict = remove_prefix(state_dict['state_dict'], 'module.')
        else:
            new_state_dict = state_dict['state_dict']
        net.load_state_dict(new_state_dict)
        start_epoch = int(state_dict['epoch']) + 1
        if args.amp:
            amp.load_state_dict(state_dict['amp'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume_net, int(state_dict['epoch']) + 1))

    step_index = 0
    for epoch in range(start_epoch, max_epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, priors, step_index, train_loader_len, net, criterion, optimizer, epoch, args,
              ngpus_per_node)

        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.max_epoch - 1) or (epoch > int(args.max_epoch * 0.9)):
            if not args.multiprocessing_distributed or (
                    args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                if args.amp:
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                    }, save_folder + 'FaceBoxes_epoch_' + str(epoch + 1) + '.pth')
                else:
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, save_folder + 'FaceBoxes_epoch_' + str(epoch + 1) + '.pth')


def train(train_loader, priors, step_index, train_loader_len, model, criterion, optimizer, epoch, args, ngpus_per_node):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e', start_count_index=0)
    progress = ProgressMeter(
        train_loader_len,
        [batch_time, data_time, losses],
        prefix="Epoch: [{}] MaxEpoch: [{}] || ".format(epoch + 1, max_epoch))

    if args.device == 'npu':
        loc = 'npu:{}'.format(args.gpu)
    elif args.device == 'gpu':
        loc = 'cuda:{}'.format(args.gpu)
    else:
        loc = 'cpu'

    model.train()
    end = time.time()
    optimizer.zero_grad()
    steps_per_epoch = train_loader_len
    print('==========step per epoch======================', steps_per_epoch)
    dataset = VOCDetection(training_dataset, preproc(img_dim, rgb_mean), AnnotationTransform())
    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size
    stepvalues = (200 * epoch_size, 250 * epoch_size)

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if epoch in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(
            optimizer, gamma, epoch, step_index, epoch, train_loader_len)
        images = images.to(loc, non_blocking=True)
        targets = [anno.to(loc) for anno in target]
        optimizer.zero_grad()
        if i == 5 and args.device_num == 1:
            print("prof out file")
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                # forward
                out = model(images)
                # backprop
                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, priors, targets, args.gpu)
                loss = cfg['loc_weight'] * loss_l + loss_c
                # loss.backward()

                if args.amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()

            prof.export_chrome_trace("output.prof")
        else:
            # forward
            out = model(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, priors, targets, args.gpu)
            loss = cfg['loc_weight'] * loss_l + loss_c
            # loss.backward()

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()


        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # measure elapse time
        if i % args.print_freq == 0:
            if not args.multiprocessing_distributed or (
                    args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                progress.display(i)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        if batch_time.avg > 0:
            print("[npu id:", args.gpu, "]",
                  '* FPS@all {:.3f}, TIME@all {:.3f}'.format(ngpus_per_node * batch_size / batch_time.avg,
                                                             batch_time.avg))


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
            self.avg = self.sum / \
                       (self.count - self.start_count_index * self.N)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


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


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr - 1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_pytorch_train_loader(data_path, batch_size, workers=5, _worker_init_fn=None, distributed=False):
    train_dataset = VOCDetection(training_dataset, preproc(
        img_dim, rgb_mean), AnnotationTransform())

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    dataloader_fn = MultiEpochsDataLoader  # torch.utils.data.DataLoader
    train_loader = dataloader_fn(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, worker_init_fn=_worker_init_fn, pin_memory=True, sampler=train_sampler,
        collate_fn=detection_collate, drop_last=True)
    return train_loader, len(train_loader), train_sampler


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


if __name__ == '__main__':
    main()
