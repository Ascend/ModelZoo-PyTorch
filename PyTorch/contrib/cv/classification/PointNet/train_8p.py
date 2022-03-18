# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016,
# Copyright 2020 Huawei Technologies Co., Ltd
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

import os
import argparse
import torch
import torch.optim as optim
from apex import amp
import torch.nn.functional as F
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.npu
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map


def profiling(input_tensor, target, model, optimizer, opt, epoch):
    def update(model, images, target, optimizer):
        pred, trans, trans_feat = model(images)
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat, opt.device) * 0.001
        optimizer.zero_grad()
        if opt.amp_mode:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

    if opt.device == 'gpu':
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            update(model, input_tensor, target, optimizer)
    elif opt.device == 'npu':
        with torch.autograd.profiler.profile(use_npu=True) as prof:
            update(model, input_tensor, target, optimizer)
    save_path = './prof_files_8p/' + opt.device
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    prof.export_chrome_trace(save_path + "/output_{}.prof".format(epoch))


def main(gpu, opt):
    if opt.device == 'gpu':
        loc = 'cuda:{}'.format(gpu)
        opt.gpu = gpu
        torch.cuda.set_device(loc)
        opt.batchSize = int(opt.batchSize / opt.world_size)
    else:
        opt.process_device_map = device_id_to_process_device_map(opt.device_list)
        if opt.device == 'npu':
            ngpus_per_node = len(opt.process_device_map)
        else:
            ngpus_per_node = torch.cuda.device_count()

        opt.gpu = opt.process_device_map[gpu]
        loc = 'npu:{}'.format(opt.gpu)

        torch.npu.set_device(loc)
        opt.batchSize = int(opt.batchSize / ngpus_per_node)
        opt.workers = int((opt.workers + ngpus_per_node - 1) / ngpus_per_node)

    rank = opt.nr * opt.gpus + gpu
    if opt.device == 'npu':
        dist.init_process_group(backend='hccl', world_size=opt.world_size, rank=rank)
    else:
        dist.init_process_group(backend='nccl', world_size=opt.world_size, rank=rank)

    if opt.dataset_type == 'shapenet':
        dataset = ShapeNetDataset(
            root=opt.dataset,
            classification=True,
            npoints=opt.num_points)

        test_dataset = ShapeNetDataset(
            root=opt.dataset,
            classification=True,
            split='test',
            npoints=opt.num_points,
            data_augmentation=False)
    elif opt.dataset_type == 'modelnet40':
        dataset = ModelNetDataset(
            root=opt.dataset,
            npoints=opt.num_points,
            split='trainval')

        test_dataset = ModelNetDataset(
            root=opt.dataset,
            split='test',
            npoints=opt.num_points,
            data_augmentation=False)
    else:
        exit('wrong dataset type')

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=opt.world_size, rank=rank
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=opt.workers,
        drop_last=True)

    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=opt.workers,
        drop_last=True)

    num_classes = len(dataset.classes)

    model = PointNetCls(k=num_classes, feature_transform=opt.feature_transform, device=opt.device).to(loc)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    amp.register_half_function(torch, 'bmm')
    classifier, optimizer = amp.initialize(model, optimizer, opt_level='O1', loss_scale=128)
    model = DDP(model, device_ids=[opt.gpu], broadcast_buffers=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    for epoch in range(opt.nepoch):
        train_sampler.set_epoch(epoch)
        total_correct = 0
        error = 0
        train_set = 0
        opt.store_prof = True
        model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        tot_time = AverageMeter()

        end = time.time()
        for i, data in enumerate(dataloader, 0):
            data_time.update(time.time() - end)
            points, target = data
            target = target[:, 0]
            if opt.device == 'npu':
                target = target.to(torch.int32)
            points = points.transpose(2, 1)
            points, target = points.to(loc, non_blocking=True), target.to(loc, non_blocking=True)

            pred, trans, trans_feat = model(points)
            loss = F.nll_loss(pred, target)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat, opt.device) * 0.001

            error += loss.item()

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                optimizer.zero_grad()
                scaled_loss.backward()
                optimizer.step()

            current_batch_time = time.time() - end
            batch_time.update(current_batch_time)
            end = time.time()

            FPS = (opt.batchSize / current_batch_time) * opt.world_size
            if i > 1:
                if gpu == 0:
                    print("Epoch %d step %d FPS: %f" % (epoch, i, FPS))
                tot_time.update(current_batch_time)

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            total_correct += correct.item()
            train_set += points.size()[0]
        epoch_FPS = (opt.batchSize / tot_time.avg) * opt.world_size
        if gpu == 0:
            print("Epoch %d avg FPS: %f" % (epoch, epoch_FPS))
            print("Epoch %d train loss: %f train acc: %f" % (epoch, error / i, total_correct / train_set))
        with torch.no_grad():
            total_correct = 0
            test_set = 0
            error = 0
            model.eval()
            for i, data in enumerate(testdataloader, 0):
                points, target = data
                target = target[:, 0]
                if opt.device == 'npu':
                    target = target.to(torch.int32)
                points = points.transpose(2, 1)
                points, target = points.to(loc, non_blocking=True), target.to(loc, non_blocking=True)

                pred, _, _ = model(points)
                loss = F.nll_loss(pred, target)

                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                error += loss.item()
                total_correct += correct.item()
                test_set += points.size()[0]
            test_acc = total_correct / test_set
            if gpu == 0:
                print("test loss: %f test accuracy %f" % (error / i, test_acc))
        scheduler.step()
        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch}

        save_path = './checkpoints_8p/' + opt.device
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        path_checkpoint = save_path + "/checkpoint_{}_epoch.pkl".format(epoch)
        torch.save(checkpoint, path_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument(
        '--num_points', type=int, default=2500, help='input batch size')
    parser.add_argument(
        '--workers', type=int, default=32, help='number of data loading workers')
    parser.add_argument(
        '--addr', default='127.0.0.1', type=str, help='master addr')
    parser.add_argument(
        '--nepoch', type=int, default=80, help='number of epochs to train for')
    parser.add_argument('--lr', default=0.001, help='learning rate')
    parser.add_argument('--amp_mode', type=bool, default=True)
    parser.add_argument('--store_prof', type=bool, default=True)
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, default="./data/shapenetcore_partanno_segmentation_benchmark_v0",
                        help="dataset path")
    parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
    parser.add_argument('--feature_transform', type=bool, default=True, help="use feature transform")
    parser.add_argument('--device', type=str, default='gpu', help='npu or gpu')
    parser.add_argument('--device-list', default='0,1,2,3,4,5,6,7', type=str, help='device id list')
    parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes")
    parser.add_argument("--gpus", default=8, type=int, help="number of gpus per node")
    parser.add_argument("--nr", default=0, type=int, help="ranking within the nodes")
    opt = parser.parse_args()

    os.environ["MASTER_ADDR"] = opt.addr
    os.environ["MASTER_PORT"] = "29501"
    opt.world_size = opt.gpus * opt.nodes

    mp.spawn(main, args=(opt,), nprocs=opt.gpus)
