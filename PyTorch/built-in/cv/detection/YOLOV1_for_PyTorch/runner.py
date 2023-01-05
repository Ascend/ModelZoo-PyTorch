from __future__ import division

import os
import argparse
import time
import math
import random
from copy import deepcopy

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config.yolo_config import yolo_config
from data.voc import VOCDetection
from data.coco import COCODataset
from data.transforms import TrainTransforms, ColorTransforms, ValTransforms

from utils import create_labels
from utils.criterion import build_criterion
from utils.misc import detection_collate

from models.yolo import build_model

from evaluator.cocoapi_evaluator import COCOAPIEvaluator
from evaluator.vocapi_evaluator import VOCAPIEvaluator

import bug_fix
if torch.__version__ >= "1.8":
    import torch_npu
import apex
from apex import amp
from bug_fix import ModelEMA


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda.')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--img_size', type=int, default=640, help='The upper bound of warm-up')
    parser.add_argument('--multi_scale_range', nargs='+', default=[10, 20], type=int, help='lr epoch to decay')
    parser.add_argument('--max_epoch', type=int, default=200, help='The upper bound of warm-up')
    parser.add_argument('--lr_epoch', nargs='+', default=[100, 150], type=int, help='lr epoch to decay')
    parser.add_argument('--wp_epoch', type=int, default=2, help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch to train')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--num_gpu', default=1, type=int, help='Number of GPUs to train')
    parser.add_argument('--eval_epoch', type=int, default=200, help='interval between evaluations')
    parser.add_argument('--save_folder', default='weights/', type=str, help='path to save weight')

    # Optimizer & Schedule
    parser.add_argument('--optimizer', default='sgd', type=str, help='sgd, adamw')
    parser.add_argument('--lr_schedule', default='step', type=str, help='step, cos')
    parser.add_argument('--grad_clip', default=None, type=float, help='clip gradient')

    # model
    parser.add_argument('-m', '--model', default='yolov1', help='yolov1, yolov2, yolov3, yolov4, yolo_tiny, yolo_nano')
    parser.add_argument('--conf_thresh', default=0.001, type=float, help='NMS threshold')
    parser.add_argument('--nms_thresh', default=0.6, type=float, help='NMS threshold')

    # dataset
    parser.add_argument('--data_path', default='/mnt/share/ssd2/dataset', help='data path')
    parser.add_argument('-d', '--dataset', default='coco', help='coco, widerface, crowdhuman')
    
    # Loss
    parser.add_argument('--loss_obj_weight', default=1.0, type=float, help='weight of obj loss')
    parser.add_argument('--loss_cls_weight', default=1.0, type=float, help='weight of cls loss')
    parser.add_argument('--loss_reg_weight', default=1.0, type=float, help='weight of reg loss')
    parser.add_argument('--scale_loss', default='batch', type=str, help='scale loss: batch or positive samples')

    # train trick
    parser.add_argument('--no_warmup', action='store_true', default=False, help='do not use warmup')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False, help='use multi-scale trick')      
    parser.add_argument('--ema', action='store_true', default=False, help='use ema training trick')
    parser.add_argument('--mosaic', action='store_true', default=False, help='use Mosaic Augmentation trick')
    parser.add_argument('--mixup', action='store_true', default=False, help='use MixUp Augmentation trick')
    parser.add_argument('--multi_anchor', action='store_true', default=False,
                        help='use multiple anchor boxes as the positive samples')
    parser.add_argument('--center_sample', action='store_true', default=False,
                        help='use center sample for labels')
    parser.add_argument('--accumulate', type=int, default=1, help='accumulate gradient')
    parser.add_argument('--fuzzy_mode', action='store_true', default=False, help='use fuzzy compilation')

    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False, help='distributed training')
    parser.add_argument('--sybn', action='store_true', default=False, help='use sybn.')

    return parser.parse_args()


def train():
    
    if args.fuzzy_mode:
        option = {}
        option["NPU_FUZZY_COMPILE_BLACKLIST"] = "Add,Mul,Slice,Conv2D,Conv2DBackpropInput,Conv2DBackpropInputD,Conv2DBackpropFilter,BNTrainingReduce,BNTrainingUpdata,BNTrainingUpdataGrad,BNTrainingReduceGrad"
        torch.npu.set_option(option)
        torch.npu.set_compile_mode(jit_compile=False)

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '26544'

    # path to save model
    path_to_save = os.path.join(args.save_folder, args.dataset, args.model)
    os.makedirs(path_to_save, exist_ok=True)

    # set distributed
    device_id = int(os.environ['RANK_ID'])
    if args.distributed:
        dist.init_process_group(backend="hccl", world_size=8, rank=device_id)
        device = torch.device(f"npu:{device_id}")
        torch.npu.set_device(device)
    else:
        if args.cuda:
            print('use npu')
            cudnn.benchmark = True
            device = torch.device("npu")
        else:
            device = torch.device("cpu")

    # YOLO config
    cfg = yolo_config[args.model]
    train_size = val_size = args.img_size

    # dataset and evaluator
    dataset, evaluator, num_classes = build_dataset(args, train_size, val_size, device)
    # dataloader
    dataloader = build_dataloader(args, dataset, detection_collate)
    # criterioin
    criterion = build_criterion(args, cfg, num_classes)
    
    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")

    # build model
    net = build_model(args=args, 
                      cfg=cfg, 
                      device=device, 
                      num_classes=num_classes, 
                      trainable=True)
    model = net

    # SyncBatchNorm
    if args.sybn and args.cuda and args.num_gpu > 1:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.to(device).train()
    
    ema = ModelEMA(model) if args.ema else None

    # optimizer setup
    base_lr = args.lr
    tmp_lr = args.lr
    if args.optimizer == 'sgd':
        print('use SGD with momentum ...')
        optimizer = apex.optimizers.NpuFusedSGD(model.parameters(), lr=tmp_lr,
                                                momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'adamw':
        print('use AdamW ...')
        optimizer = optim.AdamW(model.parameters(), 
                                lr=tmp_lr, 
                                weight_decay=5e-4)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale=128.0, combine_grad=True)

    # DDP
    if args.distributed and args.num_gpu > 1:
        print('using DDP ...')
        model = DDP(model, device_ids=[device_id], broadcast_buffers=False)

    batch_size = args.batch_size
    epoch_size = len(dataset) // (batch_size * args.num_gpu)
    best_map = -100.
    warmup = not args.no_warmup

    t0 = time.time()
    # start training loop
    for epoch in range(args.start_epoch, args.max_epoch):
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)            

        # use step lr decay
        if args.lr_schedule == 'step':
            if epoch in args.lr_epoch:
                tmp_lr = tmp_lr * 0.1
                set_lr(optimizer, tmp_lr)
        # use cos lr decay
        elif args.lr_schedule == 'cos' and not warmup:
            T_max = args.max_epoch - 15
            lr_min = base_lr * 0.1 * 0.1
            if epoch > T_max:
                # Cos decay is done
                print('Cosine annealing is over !!')
                args.lr_schedule == None
                tmp_lr = lr_min
                set_lr(optimizer, tmp_lr)
            else:
                tmp_lr = lr_min + 0.5*(base_lr - lr_min)*(1 + math.cos(math.pi*epoch / T_max))
                set_lr(optimizer, tmp_lr)

        # train one epoch
        pre_flag = False
        for iter_i, (images, targets) in enumerate(dataloader):
            if not pre_flag:
                if iter_i >= 5:
                    start_time = time.time()
                    pre_flag = True
            ni = iter_i + epoch * epoch_size
            # warmup
            if epoch < args.wp_epoch and warmup:
                nw = args.wp_epoch * epoch_size
                tmp_lr = base_lr * pow(ni / nw, 4)
                set_lr(optimizer, tmp_lr)

            elif epoch == args.wp_epoch and iter_i == 0 and warmup:
                # warmup is over
                print('Warmup is over !!')
                warmup = False
                tmp_lr = base_lr
                set_lr(optimizer, tmp_lr)

            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # randomly choose a new size
                r = args.multi_scale_range
                train_size = random.randint(r[0], r[1]) * 32
                if args.distributed:
                    model.module.set_grid(train_size)
                else:
                    model.set_grid(train_size)
            if args.multi_scale:
                # interpolate
                images = torch.nn.functional.interpolate(
                                    input=images, 
                                    size=train_size, 
                                    mode='bilinear', 
                                    align_corners=False)

            targets = [label.tolist() for label in targets]
            # make labels
            targets = create_labels.gt_creator(
                                    img_size=train_size, 
                                    strides=net.stride, 
                                    label_lists=targets, 
                                    anchor_size=cfg["anchor_size"], 
                                    multi_anchor=args.multi_anchor,
                                    center_sample=args.center_sample)
            
            # to device
            images = images.to(device)
            targets = targets.to(device)

            # inference
            pred_obj, pred_cls, pred_iou, targets = model(images, targets=targets)

            # compute loss
            loss_obj, loss_cls, loss_reg, total_loss = criterion(pred_obj, pred_cls, pred_iou, targets)

            # check loss
            if torch.isnan(total_loss):
                continue

            loss_dict = dict(
                loss_obj=loss_obj,
                loss_cls=loss_cls,
                loss_reg=loss_reg,
                total_loss=total_loss
            )
            total_loss = total_loss / args.accumulate

            # Backward and Optimize
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            
            if ni % args.accumulate == 0:
                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                # ema
                if args.ema:
                    x = torch.tensor([1.]).to(device)
                    if device.type == "npu":
                        params_fp32_fused = optimizer.get_model_combined_params()
                        ema.update(model, x, params_fp32_fused[0])
                    else:
                        ema.update(model, x)

            # display
            if iter_i % 100 == 0:
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f][Loss: obj %.2f || cls %.2f || reg %.2f || size %d || time: %.2f]'
                        % (epoch+1, 
                           args.max_epoch, 
                           iter_i, 
                           epoch_size, 
                           tmp_lr,
                           loss_dict['loss_obj'].item(), 
                           loss_dict['loss_cls'].item(), 
                           loss_dict['loss_reg'].item(), 
                           train_size, 
                           t1-t0),
                        flush=True)

                t0 = time.time()
        end_time = time.time()
        time_avg = (end_time - start_time) / (iter_i - 5)
        FPS = args.batch_size * args.num_gpu / time_avg
        print("FPS: %.2f" % FPS)

        # evaluation
        if (epoch + 1) % args.eval_epoch == 0 or (epoch + 1) == args.max_epoch:
            if evaluator is None:
                print('No evaluator ...')
                print('Saving state, epoch:', epoch + 1)
                torch.save(model_eval.state_dict(), os.path.join(path_to_save, 
                            args.model + '_' + repr(epoch + 1) + '.pth'))  
                print('Keep training ...')
            else:
                print('eval ...')
                # check ema
                if args.ema:
                    model_eval = ema.ema
                else:
                    model_eval = model.module if args.distributed else model

                # set eval mode
                model_eval.trainable = False
                model_eval.set_grid(val_size)
                model_eval.eval()

                if (args.distributed and device_id == 0) or (not args.distributed):
                    # evaluate
                    evaluator.evaluate(model_eval)

                    cur_map = evaluator.map
                    if cur_map > best_map:
                        # update best-map
                        best_map = cur_map
                        # save model
                        print('Saving state, epoch:', epoch + 1)
                        torch.save(model_eval.state_dict(), os.path.join(path_to_save, 
                                    args.model + '_' + repr(epoch + 1) + '_' + str(round(best_map*100, 2)) + '.pth'))  
                if epoch + 1 == args.max_epoch:
                    exit()
                if args.distributed:
                    # wait for all processes to synchronize
                    dist.barrier()

                # set train mode.
                model_eval.trainable = True
                model_eval.set_grid(train_size)
                model_eval.train()

        # close mosaic augmentation
        if args.mosaic and args.max_epoch - epoch == 15:
            print('close Mosaic Augmentation ...')
            dataloader.dataset.mosaic = False
        # close mixup augmentation
        if args.mixup and args.max_epoch - epoch == 15:
            print('close Mixup Augmentation ...')
            dataloader.dataset.mixup = False


def build_dataset(args, train_size, val_size, device):
    if args.dataset == 'voc':
        data_dir = os.path.join(args.data_path, 'VOCdevkit')
        num_classes = 20
        dataset = VOCDetection(
                        data_dir=data_dir,
                        img_size=train_size,
                        transform=TrainTransforms(train_size),
                        color_augment=ColorTransforms(train_size),
                        mosaic=args.mosaic,
                        mixup=args.mixup)

        evaluator = VOCAPIEvaluator(
                        data_dir=data_dir,
                        img_size=val_size,
                        device=device,
                        transform=ValTransforms(val_size))

    elif args.dataset == 'coco':
        data_dir = os.path.join(args.data_path, 'coco')
        num_classes = 80
        dataset = COCODataset(
                    data_dir=data_dir,
                    img_size=train_size,
                    image_set='train2017',
                    transform=TrainTransforms(train_size),
                    color_augment=ColorTransforms(train_size),
                    mosaic=args.mosaic,
                    mixup=args.mixup)

        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=val_size,
                        device=device,
                        transform=ValTransforms(val_size)
                        )
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    return dataset, evaluator, num_classes


def build_dataloader(args, dataset, collate_fn=None):
    # distributed
    if args.distributed and args.num_gpu > 1:
        # dataloader
        dataloader = torch.utils.data.DataLoader(
                        dataset=dataset, 
                        batch_size=args.batch_size, 
                        collate_fn=collate_fn,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        sampler=torch.utils.data.distributed.DistributedSampler(dataset)
                        )

    else:
        # dataloader
        dataloader = torch.utils.data.DataLoader(
                        dataset=dataset, 
                        shuffle=True,
                        batch_size=args.batch_size, 
                        collate_fn=collate_fn,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )
    return dataloader


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    args = parse_args()
    train()
