# -*- coding: utf-8 -*-
"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import argparse
import yaml
import os
import time
import crnn
import utils
import torch
import torch.nn.parallel
from torch.utils.data import DataLoader
from apex import amp
from easydict import EasyDict as edict


def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")
    parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)
    parser.add_argument('--npu', help='npu id', type=str)
    args = parser.parse_args()
    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)
    return config


def main():
    # load config
    config = parse_arg()
    print('config is: ', config)

    # seed everything
    utils.seed_everything()

    # construct face related neural networks
    model = crnn.get_crnn(config)

    # get device
    device = torch.device("npu:{}".format(config.DEVICE_ID))
    # device = 'npu:{}'.format(config.DEVICE_ID)
    torch.npu.set_device(device)
    model = model.to(device)

    # define loss function
    criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)

    best_acc = 0.5
    last_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = utils.get_optimizer(config, model)
    print(optimizer)

    if config.TRAIN.AMP:
        print("=> use amp, level is", config.TRAIN.OPT_LEVEL)
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.TRAIN.OPT_LEVEL,
                                          loss_scale=config.TRAIN.LOSS_SCALE)

    if config.TRAIN.RESUME.IS_RESUME:
        model_state_file = config.TRAIN.RESUME.FILE
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location=device)
        if 'state_dict' in checkpoint.keys():
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            best_acc = checkpoint['best_acc']
            optimizer.load_state_dict(checkpoint['optimizer'])
            if config.TRAIN.AMP:
                amp.load_state_dict(checkpoint['amp'])
        else:
            model.load_state_dict(checkpoint)

    # utils.model_info(model)
    train_dataset = utils.lmdbDataset(config, is_train=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        collate_fn=utils.alignCollate(32, 100),
        pin_memory=config.PIN_MEMORY,
        drop_last=config.DROP_LAST
    )

    val_dataset = utils.lmdbDataset(config, is_train=False, transform=utils.resizeNormalize((100, 32)))
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    checkpoint_dir, log_dir = utils.create_output_folder(config)
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        train(config, train_loader, train_dataset, converter, model, criterion, optimizer, device, epoch)
        acc = validate(config, val_loader, val_dataset, converter, model, criterion, device, epoch)
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        print("is best:", is_best)
        print("best acc is:", best_acc)
        if config.TRAIN.AMP:
            torch.save(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict(),
                }, os.path.join(checkpoint_dir, "checkpoint_{}_acc_{:.4f}.pth".format(epoch, acc))
            )
        else:
            torch.save(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(checkpoint_dir, "checkpoint_{}_acc_{:.4f}.pth".format(epoch, acc))
            )



def train(config, train_loader, dataset, converter, model, criterion, optimizer, device, epoch):
    utils.seed_everything()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    model.train()
    end = time.time()
    for i, (inp, idx) in enumerate(train_loader):
        data_time.update((time.time() - end) * 1000)
        labels = idx
        inp = inp.to(device)
        preds = model(inp)
        batch_size = inp.size(0)
        text, length = converter.encode(labels)
        # timestep * batchsize
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        text = text.to(device)
        length = length.to(device)
        preds_size = preds_size.to(device)
        loss = criterion(preds, text, preds_size, length)
        optimizer.zero_grad()
        if config.TRAIN.AMP:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        losses.update(loss.item(), inp.size(0))
        if i == 9:
            batch_time.reset()
            data_time.reset()
        batch_time.update((time.time() - end) * 1000)
        fps = (config.TRAIN.BATCH_SIZE_PER_GPU / batch_time.val) * 1000
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}ms ({batch_time.avg:.3f}ms)\t' \
                  'Data {data_time.val:.3f}ms ({data_time.avg:.3f}ms)\t' \
                  'Fps {fps:.3f}\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, fps=fps, loss=losses)
            print(msg)
        end = time.time()
    print(' * FPS@all {:.3f}'.format(config.TRAIN.BATCH_SIZE_PER_GPU * 1000 / batch_time.avg))


def validate(config, val_loader, dataset, converter, model, criterion, device, epoch):
    losses = utils.AverageMeter()
    model.eval()
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for i, (inp, idx) in enumerate(val_loader):
            labels = idx
            inp = inp.to(device)
            preds = model(inp)
            batch_size = inp.size(0)
            n_total = n_total + batch_size
            text, length = converter.encode(labels)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            text = text.to(device)
            length = length.to(device)
            preds_size = preds_size.to(device)
            loss = criterion(preds, text, preds_size, length)
            losses.update(loss.item(), inp.size(0))
            _, preds = preds.max(2)
            preds = preds.int()
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, labels):
                if pred == target:
                    n_correct += 1
            if (i + 1) % config.PRINT_FREQ == 0:
                print('Epoch: [{0}][{1}/{2}]'.format(epoch, i, len(val_loader)))
    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:config.TEST.NUM_TEST_DISP]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
    print(n_correct)
    print(n_total)
    accuracy = n_correct / float(n_total)
    print('Test loss: {:.4f}, accuracy: {:.4f}'.format(losses.avg, accuracy))
    return accuracy


if __name__ == '__main__':
    main()
