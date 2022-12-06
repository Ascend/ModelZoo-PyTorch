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

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import numpy as np
import pandas as pd
import os
import cv2
import pickle
import lmdb
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter

import apex
from apex import amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# from apex.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

if torch.__version__ >= '1.8':
    import torch_npu
import random
import time

from .config import config
from .alexnet import SiameseAlexNet, _create_gt_mask, Criterion
from .dataset import ImagnetVIDDataset
from .custom_transforms import Normalize, ToTensor, RandomStretch, RandomCrop, CenterCrop, RandomBlur, ColorAug


def train(data_dir, workers, epochs):
    # 1p
    CALCULATE_DEVICE = "npu:0"
    torch.npu.set_device(CALCULATE_DEVICE)
    # set random seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    # loading meta data
    meta_data_path = os.path.join(data_dir, "meta_data.pkl")
    meta_data = pickle.load(open(meta_data_path, 'rb'))
    all_videos = [x[0] for x in meta_data]  # name of seqs
    
    # split train/valid dataset using train_test_split
    train_videos, valid_videos = train_test_split(all_videos, test_size=1-config.train_ratio, random_state=config.seed)

    # define transforms  
    random_crop_size = config.instance_size - 2 * config.total_stride
    # size after Curation is 255*255, now crop (255-2*8)*(255-2*8) for search, crop 127*127 for exemplar
    train_z_transforms = transforms.Compose([  # train_z: exemplar
        RandomStretch(),
        CenterCrop((config.exemplar_size, config.exemplar_size)),
        ToTensor()
    ])
    train_x_transforms = transforms.Compose([  # train_x: search
        RandomStretch(),
        RandomCrop((random_crop_size, random_crop_size), config.max_translate),  # max translation of random shift
        ToTensor()
    ])
    valid_z_transforms = transforms.Compose([
        CenterCrop((config.exemplar_size, config.exemplar_size)),  # valid don't have RandomStretch
        ToTensor()
    ])
    valid_x_transforms = transforms.Compose([ToTensor()])
    
    db = lmdb.open(data_dir+'.lmdb', readonly=True, map_size=int(50e9))
    
    # create dataset
    train_dataset = ImagnetVIDDataset(db, train_videos, data_dir, train_z_transforms, train_x_transforms)
    valid_dataset = ImagnetVIDDataset(db, valid_videos, data_dir, valid_z_transforms,
                                      valid_x_transforms, training=False)

    # create dataloader
    trainloader = DataLoader(train_dataset, batch_size=config.train_batch_size,
                             shuffle=True, pin_memory=False,  num_workers=workers, drop_last=True)
    validloader = DataLoader(valid_dataset, batch_size=config.valid_batch_size,
                             shuffle=False, pin_memory=False, num_workers=workers, drop_last=True)
    
    # create summary writer tensorboardX
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)

    summary_writer = SummaryWriter(config.log_dir)

    # start training
    model = SiameseAlexNet()  # __init__

    # produce gt and weight
    # train_response_sz=15
    gt, weight = _create_gt_mask((config.train_response_sz, config.train_response_sz))
    train_gt = torch.from_numpy(gt).npu()
    train_weight = torch.from_numpy(weight).npu()  # train_gt.dtype=torch.float32
    # response_sz=17
    gt, weight = _create_gt_mask((config.response_sz, config.response_sz))
    valid_gt = torch.from_numpy(gt).npu()
    valid_weight = torch.from_numpy(weight).npu()

    model.init_weights()

    model = model.npu()
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,
    #                             momentum=config.momentum, weight_decay=config.weight_decay)
    optimizer = apex.optimizers.NpuFusedSGD(model.parameters(), lr=config.lr,
                                            momentum=config.momentum, weight_decay=config.weight_decay)

    # loss function
    criterion = Criterion().npu()

    # using apex
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=128.0)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=128.0, combine_grad=True)

    # lr policy
    scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    for epoch in range(epochs+1):
        # last epoch to generate prof
        if epoch == epochs:
            # generate prof
            model.train()
            for i, data in enumerate(trainloader):
                exemplar_imgs, instance_imgs = data
                exemplar_var, instance_var = Variable(exemplar_imgs.npu()), Variable(instance_imgs.npu())
                if i < 10:
                    optimizer.zero_grad()
                    outputs = model([exemplar_var, instance_var])
                    loss = criterion(outputs, train_gt, train_weight)
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    optimizer.step()
                else:
                    with torch.autograd.profiler.profile(use_npu=True) as prof:
                        optimizer.zero_grad()
                        outputs = model([exemplar_var, instance_var])
                        loss = criterion(outputs, train_gt, train_weight)
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                        optimizer.step()
                    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                    prof.export_chrome_trace("test/output/output.prof")
                    break
            break
        # normal train
        batch_time = AverageMeter('Time', ':6.3f')
        train_loss = []
        model.train()

        end = time.time()
        for i, data in enumerate(tqdm(trainloader)):
            exemplar_imgs, instance_imgs = data
            exemplar_var, instance_var = Variable(exemplar_imgs.npu()), Variable(instance_imgs.npu())

            optimizer.zero_grad()

            outputs = model([exemplar_var, instance_var])  # [batch, 1, 15, 15]

            loss = criterion(outputs, train_gt, train_weight)
            # using apex
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            # without apex
            # loss.backward()
            # calculate time
            optimizer.step()  # update parameter
            cost_time = time.time() - end

            batch_time.update(cost_time)

            step = epoch * len(trainloader) + i
            summary_writer.add_scalar('train/loss', loss.data, step)
            train_loss.append(loss.data)
            end = time.time()
            
        train_loss = torch.mean(torch.stack(train_loss))
        valid_loss = []
        model.eval()  # test mode
        for i, data in enumerate(tqdm(validloader)):
            exemplar_imgs, instance_imgs = data
            exemplar_var, instance_var = Variable(exemplar_imgs.npu()), Variable(instance_imgs.npu())

            outputs = model((exemplar_var, instance_var))
            loss = F.binary_cross_entropy_with_logits(outputs, valid_gt, valid_weight,
                                                      reduction='sum') / config.valid_batch_size  # normalize the batch
            valid_loss.append(loss.data)
        valid_loss = torch.mean(torch.stack(valid_loss))

        print("EPOCH %d valid_loss : %.4f, train_loss : %.4f" % (epoch, valid_loss, train_loss))
        summary_writer.add_scalar('valid/loss', valid_loss, (epoch + 1) * len(trainloader))

        if epochs == 1:
            print("Performance Test: ", "batch_size:", config.train_batch_size, 'Time: {:.3f}'.format(batch_time.avg),
                  '* FPS@all {:.3f}'.format(config.train_batch_size / batch_time.avg))
        else:
            torch.save(model.cpu().state_dict(), "./models/final/siamfc_{}.pth".format(epoch + 1))

        model.npu()
        scheduler.step()  # adjust lr


def train_dist(args):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '11223'

    data_dir = args.data
    # 8p
    args.device = torch.device("npu:%d" % args.local_rank)
    dist.init_process_group(backend='hccl', world_size=8, rank=args.local_rank)
    args.is_master = args.local_rank == 0

    # set random seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    # loading meta data
    meta_data_path = os.path.join(data_dir, "meta_data.pkl")  #
    meta_data = pickle.load(open(meta_data_path, 'rb'))
    all_videos = [x[0] for x in meta_data]

    # get train dataset
    train_videos, _ = train_test_split(all_videos, test_size=1 - config.train_ratio, random_state=config.seed)

    # define transforms
    random_crop_size = config.instance_size - 2 * config.total_stride
    train_z_transforms = transforms.Compose([
        RandomStretch(),
        CenterCrop((config.exemplar_size, config.exemplar_size)),
        ToTensor()
    ])
    train_x_transforms = transforms.Compose([
        RandomStretch(),
        RandomCrop((random_crop_size, random_crop_size), config.max_translate),
        ToTensor()
    ])

    db = lmdb.open(data_dir + '.lmdb', readonly=True, map_size=int(50e9))

    # create dataset
    train_dataset = ImagnetVIDDataset(db, train_videos, data_dir, train_z_transforms, train_x_transforms)

    # create distributedsampler
    train_sampler = DistributedSampler(train_dataset)

    # create dataloader
    trainloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.train_batch_size,
                             shuffle=False, pin_memory=False, num_workers=args.workers, drop_last=True)  #

    # create summary writer tensorboardX
    # only on master node
    if args.is_master:
        dist_log_dir = config.log_dir + "/dist"
        if not os.path.exists(dist_log_dir):
            os.mkdir(dist_log_dir)
        summary_writer = SummaryWriter(dist_log_dir)

    # start training
    model = SiameseAlexNet()  # __init__

    model.init_weights()

    torch.npu.set_device(args.local_rank)
    model.to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,
                                momentum=config.momentum, weight_decay=config.weight_decay)
    # optimizer = apex.optimizers.NpuFusedSGD(model.parameters(), lr=config.lr,
    #                             momentum=config.momentum, weight_decay=config.weight_decay)

    # loss function
    criterion = Criterion().to(args.device)

    # using apex
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=128.0)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=128.0, combine_grad=True)

    # adjust lr
    scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    # using ddp
    model = DDP(model, device_ids=[args.local_rank], broadcast_buffers=False)
    # model = DDP(model)

    FPSrecord = 0
    for epoch in range(args.epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        train_sampler.set_epoch(epoch)
        train_loss = []
        model.train()

        torch.npu.synchronize()
        end = time.time()
        for i, data in enumerate(trainloader):
            exemplar_imgs, instance_imgs = data
            exemplar_var, instance_var = Variable(exemplar_imgs.to(args.device)), Variable(
                instance_imgs.to(args.device))
            optimizer.zero_grad()
            outputs = model((exemplar_var, instance_var))  # [batchsize, 1, 15, 15]

            # produce gt and weight, train_response_sz=15
            gt, weight = _create_gt_mask((config.train_response_sz, config.train_response_sz))
            train_gt = torch.from_numpy(gt).to(args.device)
            train_weight = torch.from_numpy(weight).to(args.device)  # train_gt.dtype=torch.float32

            loss = criterion(outputs, train_gt, train_weight)
            # using apex
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            # without apex
            # loss.backward()
            optimizer.step()  # update parameter
            # synchronize between all devices
            torch.npu.synchronize()
            # calculate time
            cost_time = time.time() - end
            batch_time.update(cost_time)
            end = time.time()

            step = epoch * len(trainloader) + i
            if args.is_master:
                summary_writer.add_scalar('train/loss', loss.data, step)
            train_loss.append(loss.data)
        train_loss = torch.mean(torch.stack(train_loss))

        dist.all_reduce(train_loss, op=dist.reduce_op.SUM)
        if args.is_master:
            print("EPOCH %d train_loss : %.4f" % (epoch, train_loss / 8))
            FPSrecord = FPSrecord + config.train_batch_size * 8 / batch_time.avg
            print("[npu id:", args.local_rank, "]", "batch_size:", 8 * config.train_batch_size,
                  'Time: {:.3f}'.format(batch_time.avg), '* FPS@cur {:.3f}'.format(config.train_batch_size * 8 /
                                                                                   batch_time.avg))

        if args.is_master:
            if (epoch + 1) % 5 == 0:
                torch.save(model.module.state_dict(), "./models/siamfc_{}.pth".format(epoch + 1))

        scheduler.step()  # adjust lr
    if args.is_master:
        print("Training Done - [npu id:", args.local_rank, "]", "batch_size:",
              8 * config.train_batch_size, ' FPS@all {:.3f}'.format(FPSrecord / args.epoch))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', start_count_index=100):
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
