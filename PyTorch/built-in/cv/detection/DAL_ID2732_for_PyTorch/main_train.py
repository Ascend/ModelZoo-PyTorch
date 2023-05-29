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

import os
import random
import time

import numpy as np

import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from math import inf
import torch.distributed as dist
import torch.multiprocessing as mp
from apex import amp
from apex.optimizers import NpuFusedAdam
from torch_warmup_lr import WarmupLR
from datasets import UCAS_AODDataset
from utils.utils import model_info

from models.main_model import RetinaNetNPU
from models.main_anchors import StaticAnchors
from opts import parse_opts
from main_eval import evaluate
from main_utils import StaticCollector, AverageMeter, Logger

DATASETS = {
    "UCAS_AOD": UCAS_AODDataset
}


def clip_grad_norm(parameters, max_norm, optimizers, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    combine_grads = optimizers.get_optimizer_combined_grads()[0]
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
    else:
        total_norm = torch.norm(combine_grads.detach(), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        combine_grads.detach().mul_(clip_coef)
    return total_norm


def resume_model(resume_path, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])
    return model


def resume_train_utils(resume_path, begin_epoch, optimizer, scheduler):
    print('loading checkpoint {} train utils'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')

    begin_epoch = checkpoint['epoch'] + 1
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return begin_epoch, optimizer, scheduler


def get_train_utils(opt, parameters):
    assert opt.dataset in DATASETS.keys(), "Not supported dataset!"
    ds = DATASETS[opt.dataset](dataset=opt.train_path, augment=opt.augment)
    bboxes = [sample["boxes"] for sample in ds]
    max_num_boxes = max(bbox.shape[0] for bbox in bboxes)
    collater = StaticCollector(scales=opt.training_size_list, max_num_boxes=max_num_boxes)
    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(ds)
    else:
        train_sampler = None
    train_loader = data.DataLoader(
        dataset=ds,
        batch_size=opt.batch_size,
        num_workers=opt.n_threads,
        collate_fn=collater,
        shuffle=(train_sampler is None),
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler
    )
    if opt.is_master_node:
        train_step_logger = Logger(os.path.join(opt.work_dir, "{}_{}".format(opt.file_hash, "train_step.log")),
                                   ["epoch", "step", "loss_cls_val", "loss_reg_val",
                                    "loss_val", "loss_cls_avg", "loss_reg_avg", "loss_avg"])
        train_epoch_logger = Logger(os.path.join(opt.work_dir, "{}_{}".format(opt.file_hash, "train_batch.log")),
                                    ["epoch", "FPS", "loss_cls", "loss_reg", "loss"])
    else:
        train_step_logger = None
        train_epoch_logger = None

    # Optimizer
    optimizer = NpuFusedAdam(parameters, lr=opt.lr0)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[round(opt.epochs * x) for x in [0.7, 0.9]],
                                               gamma=0.1)
    scheduler = WarmupLR(scheduler, init_lr=opt.warmup_lr, num_warmup=opt.warm_epoch, warmup_strategy="cos")
    scheduler.last_epoch = opt.begin_epoch - 1
    return train_loader, train_sampler, train_step_logger, train_epoch_logger, optimizer, scheduler


def make_data_parallel(model, optimizer, opt):
    device = opt.device
    torch.npu.set_device(device)
    model.to(device)
    if opt.amp_cfg:
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt.opt_level, combine_grad=True)
    if opt.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        model = nn.DataParallel(model, device_ids=None)
    return model, optimizer


def train_epoch(epoch,
                train_loader,
                model,
                optimizer,
                original_anchors,
                opt,
                train_step_logger,
                train_epoch_logger):
    batch_time_store = AverageMeter()
    fps_store = AverageMeter()
    data_time_store = AverageMeter()
    loss_cls_store = AverageMeter()
    loss_reg_store = AverageMeter()
    end_time = time.time()
    for i, batch in enumerate(train_loader):
        data_time_store.update(time.time() - end_time)
        # start train
        model.train()
        ims, gt_boxes = batch["image"].npu(), batch["boxes"].npu()
        original_anchors_npu = original_anchors.npu()
        losses = model(ims, gt_boxes, process=epoch / opt.epochs, original_anchors=original_anchors_npu)
        loss_cls, loss_reg = losses["loss_cls"].mean(), losses["loss_reg"].mean()
        loss = loss_cls + loss_reg
        optimizer.zero_grad()
        # calculate gradient
        if opt.amp_cfg:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        clip_grad_norm(model.parameters(), 0.1, optimizer)
        optimizer.step()

        # update loop info
        batch_time_store.update(time.time() - end_time)
        end_time = time.time()
        fps_store.update(opt.world_size * opt.batch_size / batch_time_store.val)
        loss_cls_store.update(float(loss_cls))
        loss_reg_store.update(float(loss_reg))
        loss_val = loss_cls_store.val + loss_reg_store.val
        loss_avg = loss_cls_store.avg + loss_reg_store.avg
        if train_step_logger is not None:
            train_step_logger.log({
                "epoch": epoch,
                "step": i + 1,
                "loss_cls_val": loss_cls_store.val,
                "loss_reg_val": loss_reg_store.val,
                "loss_val": loss_val,
                "loss_cls_avg": loss_cls_store.avg,
                "loss_reg_avg": loss_reg_store.avg,
                "loss_avg": loss_avg,
            })

        if opt.is_master_node:
            print("Epoch: [{0}][{1}/{2}]\t"
                  "Time {batch_time_store.val:.3f} ({batch_time_store.avg:.3f})\t"
                  "Fps {fps_store.val:.3f} ({fps_store.avg:.3f})\t"
                  "Data {data_time_store.val:.3f} ({data_time_store.avg:.3f})\t"
                  "Loss_cls {loss_cls_store.val:.4f} ({loss_cls_store.avg:.4f})\t"
                  "Loss_reg {loss_reg_store.val:.4f} ({loss_reg_store.avg:.4f})\t"
                  "Loss {loss_val:.3f} ({loss_avg:.3f})".format(epoch,
                                                                i + 1,
                                                                len(train_loader),
                                                                batch_time_store=batch_time_store,
                                                                fps_store=fps_store,
                                                                data_time_store=data_time_store,
                                                                loss_cls_store=loss_cls_store,
                                                                loss_reg_store=loss_reg_store,
                                                                loss_val=loss_val,
                                                                loss_avg=loss_avg))
    if train_epoch_logger is not None:
        train_epoch_logger.log({
            "epoch": epoch,
            "FPS": fps_store.avg,
            "loss_cls": loss_cls_store.avg,
            "loss_reg": loss_reg_store.avg,
            "loss": loss_cls_store.avg + loss_reg_store.avg,

        })


def main_worker(index, opt):
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    os.environ["PYTHONHASHSEED"] = str(opt.manual_seed)
    # build folder
    last_pth = os.path.join(opt.work_dir, "{}_{}".format(opt.file_hash, "last.pth"))

    device_id = opt.process_device_map[index]
    opt.device = torch.device("npu:{}".format(device_id))
    if opt.distributed:
        opt.dist_rank = opt.dist_index * opt.npus_per_node + index
        dist.init_process_group(backend="hccl",
                                init_method=opt.dist_url,
                                world_size=opt.world_size,
                                rank=opt.dist_rank)
        opt.batch_size = int(opt.batch_size / opt.npus_per_node)
        opt.n_threads = int((opt.n_threads + opt.npus_per_node - 1) / opt.npus_per_node)
    opt.is_master_node = not opt.distributed or opt.dist_rank == 0

    model = RetinaNetNPU(backbone=opt.backbone, num_classes=opt.num_classes)
    if opt.resume_path is not None and os.path.exists(opt.resume_path):
        model = resume_model(opt.resume_path, model)

    parameters = model.parameters()
    train_loader, train_sampler, train_step_logger, train_epoch_logger, optimizer, scheduler = \
        get_train_utils(opt, parameters)
    if opt.resume_path is not None and os.path.exists(opt.resume_path):
        opt.begin_epoch, optimizer, scheduler = \
            resume_train_utils(opt.resume_path, opt.begin_epoch, optimizer, scheduler)
    model, optimizer = make_data_parallel(model, optimizer, opt)
    model_info(model, report="summary")  # "full" or "summary"
    # generate anchor
    ims_shape = np.array(opt.training_size_list, dtype="int64")
    anchor_generator = StaticAnchors(ratios=np.array([0.5, 1, 2]))
    original_anchors = anchor_generator.forward(ims_shape)

    for epoch in range(opt.begin_epoch, opt.epochs):
        if opt.is_master_node:
            print("train at epoch {}".format(epoch))
        train_epoch(epoch, train_loader, model, optimizer, original_anchors, opt, train_step_logger, train_epoch_logger)
        # Update scheduler
        scheduler.step()
        if opt.is_master_node:
            final_epoch = epoch + 1 == opt.epochs

            # Create checkpoint
            if hasattr(model, 'module'):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            chkpt = {"epoch": epoch,
                     "model": model_state_dict,
                     "optimizer": None if final_epoch else optimizer.state_dict()}

            # Save last checkpoint
            torch.save(chkpt, last_pth)
            if opt.save_interval != -1 and epoch % opt.save_interval == 0 and epoch > opt.start_save:
                save_pth = os.path.join(opt.work_dir, "{}_epoch_{}.pth".format(opt.file_hash, epoch))
                torch.save(chkpt, save_pth)
    if opt.is_master_node and opt.inference:
        # eval
        if hasattr(model, 'module'):
            eval_model = model.module
        else:
            eval_model = model
        evaluate(target_size=opt.training_size_list,
                 test_path=opt.test_path,
                 dataset=opt.dataset,
                 root_dir=opt.root_path,
                 model=eval_model,
                 num_classes=opt.num_classes)


def get_opt():
    opt = parse_opts()
    opt.process_device_map = {index: int(device_id) for index, device_id in enumerate(opt.device_list.split(","))}
    opt.training_size_list = [int(size) for size in opt.training_size.split(",")]
    return opt


def main():
    opt = get_opt()
    if not os.path.exists(opt.work_dir):
        os.mkdir(opt.work_dir)
    if opt.distributed:
        os.environ["MASTER_ADDR"] = opt.MASTER_ADDR
        os.environ["MASTER_PORT"] = opt.MASTER_PORT
        assert opt.npus_per_node <= torch.npu.device_count()
        opt.world_size = opt.npus_per_node * opt.dist_num
        mp.spawn(main_worker, nprocs=opt.npus_per_node, args=(opt,))
    else:
        opt.world_size = 1
        main_worker(0, opt)


if __name__ == "__main__":
    main()
