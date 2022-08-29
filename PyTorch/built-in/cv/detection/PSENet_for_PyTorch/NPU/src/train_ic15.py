#     Copyright [yyyy] [name of copyright owner]
#     Copyright 2020 Huawei Technologies Co., Ltd
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#

import argparse
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.npu
import torch.utils.data.distributed
from apex import amp

import apex
import models
from data_loader import IC15Loader
from metrics import runningScore
from multi_epochs_dataloader import MultiEpochsDataLoader
from util import AverageMeter



def ohem_single(score, gt_text, training_mask):
    pos_num = (int)(np.sum(gt_text > 0.5)) - (int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))

    if pos_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    neg_num = (int)(np.sum(gt_text <= 0.5))
    neg_num = (int)(min(pos_num * 3, neg_num))

    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    neg_score = score[gt_text <= 0.5]
    neg_score_sorted = np.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
    return selected_mask


def ohem_batch(scores, gt_texts, training_masks):
    scores = scores.data.cpu().numpy()
    gt_texts = gt_texts.data.cpu().numpy()
    training_masks = training_masks.data.cpu().numpy()
    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))
    selected_masks = np.concatenate(selected_masks, 0)
    selected_masks = torch.from_numpy(selected_masks).float()
    return selected_masks


def dice_loss(input, target, mask):
    input = torch.sigmoid(input)

    input = input.reshape(input.size()[0], -1)
    target = target.reshape(target.size()[0], -1)
    mask = mask.reshape(mask.size()[0], -1)

    input = input * mask
    target = target * mask

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = torch.mean(d)
    return 1 - dice_loss


def cal_text_score(texts, gt_texts, training_masks, running_metric_text):
    training_masks = training_masks.data.cpu().numpy()
    pred_text = torch.sigmoid(texts).data.cpu().numpy() * training_masks

    pred_text[pred_text <= 0.5] = 0
    pred_text[pred_text > 0.5] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text


def cal_kernel_score(kernels, gt_kernels, gt_texts, training_masks, running_metric_kernel):
    mask = (gt_texts * training_masks).data.cpu().numpy()
    kernel = kernels[:, -1, :, :]
    gt_kernel = gt_kernels[:, -1, :, :]
    pred_kernel = torch.sigmoid(kernel).data.cpu().numpy()
    pred_kernel[pred_kernel <= 0.5] = 0
    pred_kernel[pred_kernel > 0.5] = 1
    pred_kernel = (pred_kernel * mask).astype(np.int32)
    gt_kernel = gt_kernel.data.cpu().numpy()
    gt_kernel = (gt_kernel * mask).astype(np.int32)
    running_metric_kernel.update(gt_kernel, pred_kernel)
    score_kernel, _ = running_metric_kernel.get_scores()
    return score_kernel


def train(train_loader, model, criterion, optimizer, epoch,  args, npu_per_node):
    model.train()

    losses = AverageMeter()
    running_metric_text = runningScore(2)
    running_metric_kernel = runningScore(2)

    epoch_time = time.time()
    batch_time = time.time()
    for batch_idx, (imgs, gt_texts, gt_kernels, training_masks) in enumerate(train_loader):
        loc = 'npu:{}'.format(args.npu)
        imgs = imgs.to(loc, non_blocking=True)
        gt_texts = gt_texts.to(loc, non_blocking=True)
        gt_kernels = gt_kernels.to(loc, non_blocking=True)
        training_masks = training_masks.to(loc, non_blocking=True)

        outputs = model(imgs)
        texts = torch.index_select(outputs, 1, torch.tensor([0]).to(loc)).squeeze()
        kernels = torch.index_select(outputs, 1, torch.tensor([1, 2, 3, 4, 5, 6]).to(loc))

        selected_masks = ohem_batch(texts, gt_texts, training_masks)
        selected_masks = selected_masks.to(loc, non_blocking=True)

        loss_text = criterion(texts, gt_texts, selected_masks)

        loss_kernels = []
        mask0 = torch.sigmoid(texts).data.cpu().numpy()
        mask1 = training_masks.data.cpu().numpy()
        selected_masks = ((mask0 > 0.5) & (mask1 > 0.5)).astype('float32')
        selected_masks = torch.from_numpy(selected_masks).float()
        selected_masks = selected_masks.to(loc, non_blocking=True)
        for i in range(6):
            kernel_i = torch.index_select(kernels, 1, torch.tensor([i]).to(loc)).squeeze()
            gt_kernel_i = torch.index_select(gt_kernels, 1, torch.tensor([i]).to(loc)).squeeze()
            loss_kernel_i = criterion(kernel_i, gt_kernel_i, selected_masks)
            loss_kernels.append(loss_kernel_i)
        loss_kernel = sum(loss_kernels) / len(loss_kernels)

        loss = 0.7 * loss_text + 0.3 * loss_kernel
        losses.update(loss.item(), imgs.size(0))

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        score_text = cal_text_score(texts, gt_texts, training_masks, running_metric_text)
        score_kernel = cal_kernel_score(kernels, gt_kernels, gt_texts, training_masks, running_metric_kernel)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed 
                                                    and args.rank % npu_per_node == 0):
            batch_time = time.time() - batch_time
            output_log = '(epoch: {epoch:0>3d} {batch:0>2d}/{size}) | FPS: {fps:5.3f} | Loss : {lossv:.4f} | Acc_t: {acc: .4f} | IOU_t: {iou_t: .4f} | IOU_k: {iou_k: .4f}'.format(
                epoch=epoch + 1,
                batch=batch_idx + 1,
                size=len(train_loader),
                fps=npu_per_node * args.batch_size / batch_time,
                lossv=losses.val,
                acc=score_text['Mean Acc'],
                iou_t=score_text['Mean IoU'],
                iou_k=score_kernel['Mean IoU'])
            batch_time = time.time()
            print(output_log)
    epoch_time = time.time() - epoch_time

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % npu_per_node == 0):
        output_log = '{epoch:0>3d}/{n_epoch} | LR: {lr:.5f} | FPS: {fps:5.3f} | batch: {batch:.5f}s | Loss: {lossa:.4f} | Acc_t: {acc: .4f} | IOU_t: {iou_t: .4f} | IOU_k: {iou_k: .4f}'.format(
            epoch=epoch + 1,
            n_epoch=args.n_epoch,
            lr=optimizer.param_groups[0]['lr'],
            fps=npu_per_node * len(train_loader) * args.batch_size / epoch_time,
            batch=epoch_time / len(train_loader),
            lossa=losses.avg,
            acc=score_text['Mean Acc'],
            iou_t=score_text['Mean IoU'],
            iou_k=score_kernel['Mean IoU'])
        print(output_log)
        sys.stdout.flush()

    return (
        losses.avg, score_text['Mean Acc'], score_kernel['Mean Acc'], score_text['Mean IoU'], score_kernel['Mean IoU'])


def adjust_learning_rate(args, optimizer, epoch):
    global state
    if epoch in args.schedule:
        args.lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr


def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    if not os.path.isdir(checkpoint):
        os.makedirs(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


def main(npu, npu_per_node, args):
    args.npu = args.process_device_map[npu]
    print("[npu id:", args.npu, "]", "+++++++++++++++++++++++++++ before set KERNEL_NAME_ID:",
          os.environ['KERNEL_NAME_ID'])
    os.environ['KERNEL_NAME_ID'] = str(npu)
    print("[npu id:", args.npu, "]", "+++++++++++++++++++++++++++KERNEL_NAME_ID:", os.environ['KERNEL_NAME_ID'])

    if args.npu is not None:
        print("[npu id:", args.npu, "]", "Use NPU: {} for training".format(args.npu))

    if args.checkpoint == '':
        args.checkpoint = "checkpoints/ic15_%s_bs_%d_ep_%d" % (args.arch, args.batch_size, args.n_epoch)
    if args.pretrain:
        if 'synth' in args.pretrain:
            args.checkpoint += "_pretrain_synth"
        else:
            args.checkpoint += "_pretrain_ic17"
    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * npu_per_node + npu
        if args.device == 'npu':
            dist.init_process_group(backend=args.dist_backend,
                                    world_size=args.world_size, rank=args.rank)
    loc = 'npu:{}'.format(args.npu)
    torch.npu.set_device(loc)
    args.batch_size = int(args.batch_size / npu_per_node)
    args.workers = int((args.workers + npu_per_node - 1) / npu_per_node)

    print("[npu id:", args.npu, "]", "===============main_worker()=================")
    print("[npu id:", args.npu, "]", args)
    print("[npu id:", args.npu, "]", "===============main_worker()=================")

    print('checkpoint path: %s' % args.checkpoint)
    print('init lr: %.8f' % args.lr)
    print('schedule: ', args.schedule)
    sys.stdout.flush()

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    kernel_num = 7
    min_scale = 0.4
    start_epoch = 0

    my_data = IC15Loader(args=args,
                         is_transform=True,
                         img_size=args.img_size,
                         kernel_num=kernel_num,
                         min_scale=min_scale)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(my_data)
    else:
        train_sampler = None

    train_loader = MultiEpochsDataLoader(
        my_data,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True,
        sampler=train_sampler)

    print("[npu id:", args.npu, "]", "=> creating model '{}'".format(args.arch))
    if args.arch == "resnet50":
        model = models.resnet50(pretrained=True, num_classes=kernel_num)
    elif args.arch == "resnet101":
        model = models.resnet101(pretrained=True, num_classes=kernel_num)
    elif args.arch == "resnet152":
        model = models.resnet152(pretrained=True, num_classes=kernel_num)

    model = model.to(loc)

    if args.combine_sgd:
        optimizer = apex.optimizers.NpuFusedSGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=5e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=5e-4)

    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale,
                                      combine_grad=args.combine_grad)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.npu], broadcast_buffers=False)

    if args.pretrain:
        print('Using pretrained model.')
        assert os.path.isfile(args.pretrain), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.pretrain)
        model.load_state_dict(checkpoint['state_dict'])
    elif args.resume:
        print('Resuming from checkpoint.')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        amp.load_state_dict(checkpoint['amp'])
    else:
        print('Training from scratch.')
    cudnn.benchmark = True

    best_model = {'loss': 0, 'acc': 0, 'iou': 0}

    for epoch in range(start_epoch, args.n_epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(args, optimizer, epoch)

        train_loss, train_te_acc, train_ke_acc, train_te_iou, train_ke_iou = train(train_loader, model, dice_loss,
                                                                                   optimizer, epoch,
                                                                                   args, npu_per_node)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % npu_per_node == 0):
            if epoch > args.n_epoch - 6:
                best_path = f'{args.remark}_{train_loss:.4f}_{train_te_acc:.4f}_{train_ke_iou:.4f}_{train_te_iou:.4f}_{epoch}.pth.tar'
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'lr': args.lr,
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict(),
                }, checkpoint='best', filename=best_path)
                best_model['acc'] = train_te_acc
                best_model['iou'] = train_te_iou


def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--data-dir', nargs='?', type=str, default='PSENet_data',
                        help='point to the root data path of ICDAR')
    parser.add_argument('--train_data', nargs='?', type=str, default='ICDAR2015',
                        help='indicate which dataset was used, ICDAR2015 or ICDAR2017')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet50')
    parser.add_argument('--img_size', nargs='?', type=int, default=640,
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=600,
                        help='# of the epochs')
    parser.add_argument('--schedule', type=int, nargs='+', default=[200, 400],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=16,
                        help='Batch Size')
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--pretrain', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--opt-level', type=str)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=64)
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--device-list', default='0,1,2,3,4,5,6,7', type=str, help='device id list')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N NPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--device', default='npu', type=str,
                        help='npu or gpu')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--addr', default='10.136.181.127', type=str,
                        help='master addr')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--port', default='8888', type=str)
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--remark', default='', type=str,
                        help='remark. ')
    parser.add_argument('--combine_grad', action='store_true',
                        help='whether to combine grad in apex')
    parser.add_argument('--combine_sgd', action='store_true',
                        help='whether to use combined sgd instead of sgd')

    args = parser.parse_args()


    if args.seed is not None:
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.process_device_map = device_id_to_process_device_map(args.device_list)

    if args.device == 'npu':
        npu_per_node = len(args.process_device_map)
    else:
        npu_per_node = torch.cuda.device_count()

    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port
    os.environ['KERNEL_NAME_ID'] = str(0)
    print("+++++++++++++++++++++KERNEL_NAME_ID:", os.environ['KERNEL_NAME_ID'])

    args.world_size = npu_per_node * args.world_size
    mp.spawn(main, nprocs=npu_per_node, args=(npu_per_node, args))

