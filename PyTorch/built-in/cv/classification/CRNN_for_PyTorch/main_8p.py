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
import ast
import yaml
import os
import time
import crnn
import utils
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.nn.parallel
from torch.utils.data import DataLoader
from apex import amp
from easydict import EasyDict as edict

import torch.distributed as dist
import torch.utils.data.distributed
try:
    from torch_npu.utils.profiler import Profile
except ImportError:
    print("Profile not in torch_npu.utils.profiler now... Auto Profile disabled.", flush=True)
    class Profile:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def end(self):
            pass


def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")
    parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)
    parser.add_argument('--npu', help='npu id', type=int)
    parser.add_argument('--bin', type=ast.literal_eval, default=False, help='enable run time2.0 model')
    parser.add_argument('--pro', type=ast.literal_eval, default=False, help='enable control steps number')
    parser.add_argument('--training_debug', type=ast.literal_eval,
                        default=False, help='enable control train_model is debug')
    parser.add_argument('--training_type', type=ast.literal_eval,
                        default=False, help="enable control train_model is 'GE' or 'CANN'")
    parser.add_argument('--profiling', type=str, default='NONE',help='choose profiling way--CANN,GE,NONE')
    parser.add_argument('--max_step', default=10, type=int, help='start_step')
    parser.add_argument('--start_step', default=0, type=int, help='start_step')
    parser.add_argument('--stop_step', default=1000, type=int,help='stop_step')
    parser.add_argument('--ND', type=ast.literal_eval, default=False, help="enable nd compile")
    args = parser.parse_args()
    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f)
        config = edict(config)
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)
    return config, args


def main():
    # load config
    config, args = parse_arg()
    if args.ND:
        print('***********allow_internal_format = False*******************')
        torch.npu.config.allow_internal_format = False
    npu = args.npu
    if args.bin:
        print('enable run time2.0 model now!')
        torch.npu.set_compile_mode(jit_compile=False)

    print('config is: ', config)

    # seed everything
    utils.seed_everything()

    if config.DISTRIBUTED.DIST_URL == "env://" and config.DISTRIBUTED.WORLD_SIZE == -1:
        config.DISTRIBUTED.WORLD_SIZE = int(os.environ["WORLD_SIZE"])

    npus_per_node = int(os.environ["RANK_SIZE"])

    if config.DISTRIBUTED.MULTIPROCESSING_DISTRIBUTED:
        # Since we have ngpus_per_node processes per node, the total world_size needs to be adjusted accordingly
        # world_size means nums of all devices or nums of processes
        config.DISTRIBUTED.WORLD_SIZE = npus_per_node * config.DISTRIBUTED.WORLD_SIZE

    if npu is not None:
        print("[npu id:", npu, "]", "Use NPU: {} for training".format(npu))

    if config.DISTRIBUTED.DIST_URL == "env://" and config.DISTRIBUTED.RANK == -1:
        config.DISTRIBUTED.RANK = int(os.environ["RANK"])
    if config.DISTRIBUTED.MULTIPROCESSING_DISTRIBUTED:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        config.DISTRIBUTED.RANK = config.DISTRIBUTED.RANK * npus_per_node + npu

    print("rank:", config.DISTRIBUTED.RANK)
    distributed = config.DISTRIBUTED.WORLD_SIZE > 1 or config.DISTRIBUTED.MULTIPROCESSING_DISTRIBUTED
    if distributed:
        os.environ['MASTER_ADDR'] = config.DISTRIBUTED.ADDR
        os.environ['MASTER_PORT'] = '29501'
        dist.init_process_group(backend=config.DISTRIBUTED.DIST_BACKEND,  # init_method=cfg.dist_url,
                                world_size=config.DISTRIBUTED.WORLD_SIZE, rank=config.DISTRIBUTED.RANK)

    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of NPUs we have
    if config.DISTRIBUTED.WORLD_SIZE >= 16:
        config.TRAIN.BATCH_SIZE_PER_GPU = int(config.TRAIN.BATCH_SIZE_PER_GPU / config.DISTRIBUTED.WORLD_SIZE)
    else:
        config.TRAIN.BATCH_SIZE_PER_GPU = int(config.TRAIN.BATCH_SIZE_PER_GPU / npus_per_node)
    config.WORKERS = int((config.WORKERS + npus_per_node - 1) / npus_per_node)
    print("batchsize:", config.TRAIN.BATCH_SIZE_PER_GPU)
    print("workers:", config.WORKERS)

    # construct face related neural networks
    model = crnn.get_crnn(config)

    # get device
    device = torch.device("npu:{}".format(npu))
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
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=(train_sampler is None),
        num_workers=config.WORKERS,
        # H,W
        collate_fn=utils.alignCollate(32, 100),
        pin_memory=config.PIN_MEMORY,
        sampler=train_sampler,
        drop_last=config.DROP_LAST
    )

    # W,H
    val_dataset = utils.lmdbDataset(config, is_train=False, transform=utils.resizeNormalize((100, 32)))
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    # Wrap the model, data parallel
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[npu], broadcast_buffers=False)

    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    if config.DISTRIBUTED.RANK % npus_per_node == 0:
        checkpoint_dir, log_dir = utils.create_output_folder(config)
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        train(config, train_loader, train_dataset, converter, model, criterion, optimizer, device, epoch, npus_per_node,
              npu)
        acc = validate(config, val_loader, val_dataset, converter, model, criterion, device, epoch)
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        print("is best:", is_best)
        print("best acc is:", best_acc)
        if config.DISTRIBUTED.RANK % npus_per_node == 0:
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

class NoProfiling():
    def __enter__(self):
        ...
    def __exit__(self, exc_type, exc_val, exc_tb):
        ...

def train(config, train_loader, dataset, converter, model, criterion, optimizer, device, epoch, npus_per_node, npu):
    _, args = parse_arg()
    utils.seed_everything()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    model.train()
    end = time.time()
    num_steps = 0
    distributed = npus_per_node > 1
    profiler = Profile(start_step=int(os.getenv("PROFILE_START_STEP", 10)),
                       profile_type=os.getenv("PROFILE_TYPE"))

    for i, (inp, idx) in enumerate(train_loader):
        if not distributed and args.training_debug and i >= args.max_step:
            break
        if not distributed and args.pro and i >= 1000:
            break
        if not distributed and args.training_type and args.num_steps > args.stop_step:
            import sys
            sys.exit()
        profiler.start()
        data_time.update((time.time() - end) * 1000)
        labels = idx
        inp = inp.to(device)
        preds = model(inp)
        batch_size = inp.size(0)
        text, length = converter.encode(labels)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)  # timestep * batchsize
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
        profiler.end()
        losses.update(loss.item(), inp.size(0))
        if i == 9:
            batch_time.reset()
            data_time.reset()
        batch_time.update((time.time() - end) * 1000)
        fps = npus_per_node * config.TRAIN.BATCH_SIZE_PER_GPU * 1000 / batch_time.val
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
    print("[npu id:", npu, "]",
          ' * FPS@all {:.3f}'.format(npus_per_node * config.TRAIN.BATCH_SIZE_PER_GPU * 1000 / batch_time.avg))


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
