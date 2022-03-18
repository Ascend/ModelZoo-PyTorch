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

import argparse
import os
import random
import time
import torch
import torch.nn.parallel
from apex import amp
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=64, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=80, help='number of epochs to train for')

parser.add_argument('--device', type=str, default='gpu')
parser.add_argument('--amp_mode', type=bool, default=True)
parser.add_argument('--store_prof', type=bool, default=True)
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, default="./data/shapenetcore_partanno_segmentation_benchmark_v0",
                    help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', type=bool, default=True, help="use feature transform")

opt = parser.parse_args()

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)

random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


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


device = None
if opt.device == 'gpu':
    device = "cuda:0"
    torch.cuda.set_device(device)
else:
    device = "npu:0"
    torch.npu.set_device(device)


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
    save_path = './prof_files/' + opt.device
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    prof.export_chrome_trace(save_path + "/output_{}.prof".format(epoch))


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

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

num_classes = len(dataset.classes)

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform, device=opt.device)
classifier = classifier.to(device)
if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model)['model_state_dict'])

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
amp.register_half_function(torch, 'bmm')
classifier, optimizer = amp.initialize(classifier, optimizer, opt_level='O1', loss_scale=128)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    opt.store_prof = True
    total_correct = 0
    error = 0
    train_set = 0
    classifier = classifier.train()
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
        points, target = points.to(device, non_blocking=True), target.to(device, non_blocking=True)

        if i > 4 and opt.store_prof:
            profiling(points, target, classifier, optimizer, opt, epoch)
            opt.store_prof = False
        pred, trans, trans_feat = classifier(points)
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
        FPS = opt.batchSize / current_batch_time

        if i > 4:
            print("Epoch %d step %d FPS: %f" % (epoch, i, FPS))
            tot_time.update(current_batch_time)

        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        train_set += points.size()[0]
    epoch_FPS = opt.batchSize / tot_time.avg
    print("Epoch %d avg FPS: %f" % (epoch, epoch_FPS))
    print("Epoch %d train loss: %f train acc: %f" % (epoch, error / i, total_correct / train_set))
    with torch.no_grad():
        total_correct = 0
        test_set = 0
        error = 0
        classifier = classifier.eval()
        for i, data in enumerate(testdataloader, 0):
            points, target = data
            target = target[:, 0]
            if opt.device == 'npu':
                target = target.to(torch.int32)
            points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)
            pred, _, _ = classifier(points)
            loss = F.nll_loss(pred, target)

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            error += loss.item()
            total_correct += correct.item()
            test_set += points.size()[0]
        test_acc = total_correct / test_set
        print("test loss: %f test accuracy %f" % (error / i, test_acc))
    scheduler.step()
    checkpoint = {"model_state_dict": classifier.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "epoch": epoch}

    save_path = './checkpoints/' + opt.device
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    path_checkpoint = save_path + "/checkpoint_{}_epoch.pkl".format(epoch)
    torch.save(checkpoint, path_checkpoint)
