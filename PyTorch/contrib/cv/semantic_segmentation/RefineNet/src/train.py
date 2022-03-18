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
# ============================================================================
import os
import re
import sys
from tqdm import tqdm
from time import time
sys.path.append('./')
# general libs
import logging
import numpy as np

# pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# densetorch wrapper
import densetorch as dt

# configuration for light-weight refinenet
from arguments import get_arguments
from data import get_datasets, get_transforms
from network import get_segmenter
from optimisers import get_optimisers, get_lr_schedulers
from apex import amp
import torch.multiprocessing as mp

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', start_count_index=5):
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



logger = logging.getLogger(__name__)
def setup_network(args, device):
    logger = logging.getLogger(__name__)
    segmenter = get_segmenter(
        enc_backbone=args.enc_backbone,
        enc_pretrained=args.enc_pretrained,
        num_classes=args.num_classes,
    ).to(device)
    print(
        " Loaded Segmenter {}, ImageNet-Pre-Trained={}, #PARAMS={:3.2f}M".format(
            args.enc_backbone,
            args.enc_pretrained,
            dt.misc.compute_params(segmenter) / 1e6,
        )
    )
    training_loss = nn.CrossEntropyLoss(ignore_index=args.ignore_label).to(device)
    validation_loss = dt.engine.MeanIoU(num_classes=args.num_classes)
    return segmenter, training_loss, validation_loss


def setup_checkpoint_and_maybe_restore(args, model, optimisers, schedulers):
    saver = dt.misc.Saver(
        args=vars(args),
        ckpt_dir=args.ckpt_dir,
        best_val=0,
        condition=lambda x, y: x > y,
    )  # keep checkpoint with the best validation score
    (
        epoch_start,
        _,
        state_dict,
        _,
        _,
    ) = saver.maybe_load(
        ckpt_path=args.ckpt_path,
        keys_to_load=["epoch", "best_val", "model", "optimisers", "schedulers"],
    )
    epoch_start = 0
    if state_dict is None:
        if len(args.ckpt_path)>3:
            print("can't find", args.ckpt_path)
            exit()
        return saver, epoch_start
    print("load pretrained from", args.ckpt_path)
    
    is_module_model_dict = list(model.state_dict().keys())[0].startswith("module")
    is_module_state_dict = list(state_dict.keys())[0].startswith("module")
    if is_module_model_dict and is_module_state_dict:
        pass
    elif is_module_model_dict:
        state_dict = {"module." + k: v for k, v in state_dict.items()}
    elif is_module_state_dict:
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    target_state_dict = model.state_dict()
    for x in target_state_dict:
        if x in state_dict and target_state_dict[x].size()==state_dict[x].size():
            target_state_dict[x] = state_dict[x]
        else:
            print(x, "shape mismatch", target_state_dict[x].size(), state_dict[x].size())

    model.load_state_dict(target_state_dict, strict=False)
    return saver, epoch_start


def setup_data_loaders(args):
    train_transforms, val_transforms = get_transforms(
        crop_size=args.crop_size,
        shorter_side=args.shorter_side,
        low_scale=args.low_scale,
        high_scale=args.high_scale,
        img_mean=args.img_mean,
        img_std=args.img_std,
        img_scale=args.img_scale,
        ignore_label=args.ignore_label,
        num_stages=args.num_stages,
        augmentations_type=args.augmentations_type,
        dataset_type=args.dataset_type,
    )
    train_sets, val_set = get_datasets(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        train_list_path=args.train_list_path,
        val_list_path=args.val_list_path,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        masks_names=("segm",),
        dataset_type=args.dataset_type,
        stage_names=args.stage_names,
        train_download=args.train_download,
        val_download=args.val_download,
    )
    train_loaders, val_loader, train_sampler = amp_get_loaders(
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        train_set=train_sets,
        val_set=val_set,
        num_stages=args.num_stages,
        distributed=args.distributed,
    )
    return train_loaders, val_loader, train_sampler


def setup_optimisers_and_schedulers(args, model):
    optimisers = get_optimisers(
        model=model,
        enc_optim_type=args.enc_optim_type,
        enc_lr=args.enc_lr,
        enc_weight_decay=args.enc_weight_decay,
        enc_momentum=args.enc_momentum,
        dec_optim_type=args.dec_optim_type,
        dec_lr=args.dec_lr,
        dec_weight_decay=args.dec_weight_decay,
        dec_momentum=args.dec_momentum,
    )
    schedulers = get_lr_schedulers(
        enc_optim=optimisers[0],
        dec_optim=optimisers[1],
        enc_lr_gamma=args.enc_lr_gamma,
        dec_lr_gamma=args.dec_lr_gamma,
        enc_scheduler_type=args.enc_scheduler_type,
        dec_scheduler_type=args.dec_scheduler_type,
        epochs_per_stage=args.epochs_per_stage,
    )
    return optimisers, schedulers

def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id
    
    return process_device_map

def main():
    args = get_arguments()
    print(args)
    os.environ['MASTER_ADDR'] = '127.0.0.1' # 可以使用当前真实ip或者'127.0.0.1'
    os.environ['MASTER_PORT'] = '29688' # 随意一个可使用的port即可

    device_list = args.device_list
    args.process_device_map = device_id_to_process_device_map(device_list)
    ngpus_per_node = len(args.process_device_map)
    if ngpus_per_node==1:
        args.distributed = False
        args.world_size = ngpus_per_node * 1
        gpu = 0
        main_worker(gpu, 1, args)
    else:
        args.distributed = True
        args.world_size = ngpus_per_node * 1
        npu = args.local_rank
        main_worker(npu, ngpus_per_node, args)



def main_worker(gpu, ngpus_per_node, args):
    args.gpu = args.process_device_map[gpu]
    device_type = args.device_type
    ## set gpu/npu
    device = '{}:{}'.format(device_type, args.gpu)
    if args.device_type=="npu": torch.npu.set_device(device)

    print("[", device_type, " id:", args.gpu, "]", "===============main_worker()=================")
    print("[", device_type, " id:", args.gpu, "]", args)
    print("[", device_type, " id:", args.gpu, "]", "===============main_worker()=================")

    args.rank = 0 * ngpus_per_node + gpu
    if args.distributed:
        if args.device_type=="npu":
            torch.distributed.init_process_group(backend='hccl', # init_method="tcp://127.0.0.1:12345",
                                            world_size=args.world_size, rank=args.rank)
        else:
            torch.distributed.init_process_group(backend='nccl', init_method="tcp://127.0.0.1:12345",
                                            world_size=args.world_size, rank=args.rank)
        torch.backends.cudnn.deterministic = True

    dt.misc.set_seed(args.random_seed)

    # Network
    segmenter, training_loss, validation_loss = setup_network(args, device=device)
    # Optimisers
    optimisers, schedulers = setup_optimisers_and_schedulers(args, model=segmenter)
    # Checkpoint
    saver, restart_epoch = setup_checkpoint_and_maybe_restore(
        args, model=segmenter, optimisers=optimisers, schedulers=schedulers,
    )
    # Calculate from which stage and which epoch to restart the training
    total_epoch = restart_epoch
    all_epochs = np.cumsum(args.epochs_per_stage)
    # APEX setting
    from apex import amp
    segmenter, [optimisers_enc, optimisers_dec] = amp.initialize(segmenter, [optimisers[0], optimisers[1]], opt_level="O2", loss_scale=1024.0, combine_grad=True)
    optimisers = [optimisers_enc, optimisers_dec]

    if args.distributed:
        segmenter = torch.nn.parallel.DistributedDataParallel(segmenter, device_ids=[args.gpu], broadcast_buffers=False)

    # # Data
    train_loaders, val_loader, train_sampler = setup_data_loaders(args)

    restart_stage = sum(restart_epoch >= all_epochs)
    if restart_stage > 0:
        restart_epoch -= all_epochs[restart_stage - 1]
    for stage in range(restart_stage, args.num_stages):
        batch_size = args.train_batch_size[stage]
        print("ngpu {:d}, BS {:d}".format(ngpus_per_node, batch_size))
        if stage > restart_stage:
            restart_epoch = 0
        for epoch in range(restart_epoch, args.epochs_per_stage[stage]):
            if args.distributed: train_sampler[stage].set_epoch(epoch)
            loss, loss_avg, time_avg = amp_train(
                model=segmenter,
                opts=optimisers,
                crits=training_loss,
                dataloader=train_loaders[stage],
                freeze_bn=args.freeze_bn[stage],
                grad_norm=args.grad_norm[stage],
                stage=stage,
                epoch=epoch,
            )
            
            print("[gpu id:", args.gpu, "]", 
                f"Training: stage {stage} epoch {epoch}",
                "Loss {:.3f} | Avg. Loss {:.3f}".format(loss, loss_avg), 
                '* FPS@all {:.3f}, TIME@all {:.3f}'.format(ngpus_per_node * batch_size / time_avg, time_avg)
            )

            total_epoch += 1
            for scheduler in schedulers:
                scheduler.step(total_epoch)
                
    vals = amp_validate(
        model=segmenter, metrics=validation_loss, dataloader=val_loader,stage=stage,epoch=epoch,
    )
    if args.gpu==0:
        saver.maybe_save(
            new_val=vals,
            dict_to_save={
                "model": segmenter.state_dict(),
                "epoch": total_epoch,
                "optimisers": [
                    optimiser.state_dict() for optimiser in optimisers
                ],
                "schedulers": [
                    scheduler.state_dict() for scheduler in schedulers
                ],
            },
        )

def maybe_cast_target_to_long(target):
    """Torch losses usually work on Long types"""
    if target.dtype == torch.uint8:
        return target.to(torch.long)
    return target

def get_input_and_targets(sample, dataloader, device):
    if isinstance(sample, dict):
        input = sample["image"].float().to(device)
        targets = [
            maybe_cast_target_to_long(sample[k].to(device))
            for k in dataloader.dataset.masks_names
        ]
    elif isinstance(sample, (tuple, list)):
        input, *targets = sample
        input = input.float().to(device)
        targets = [maybe_cast_target_to_long(target.to(device)) for target in targets]
    else:
        raise Exception(f"Sample type {type(sample)} is not supported.")
    return input, targets

def amp_train(
    model, opts, crits, dataloader, loss_coeffs=(1.0,), freeze_bn=False, grad_norm=0.0, stage=0, epoch=0
):

    model.train()
    if freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    device = next(model.parameters()).device
    opts = dt.misc.utils.make_list(opts)
    crits = dt.misc.utils.make_list(crits)
    loss_coeffs = dt.misc.utils.make_list(loss_coeffs)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_meter = AverageMeter('Loss', ':.4e', start_count_index=0)
    pbar = dataloader

    end = time()
    for idx, sample in enumerate(pbar):
        data_time.update(time() - end)
        loss = 0.0
        input, targets = get_input_and_targets(
            sample=sample, dataloader=dataloader, device=device
        )
        outputs = model(input)
        outputs = dt.misc.utils.make_list(outputs)
        for out, target, crit, loss_coeff in zip(outputs, targets, crits, loss_coeffs):
            loss += loss_coeff * crit(
                F.interpolate(
                    out, size=target.size()[1:], mode="bilinear", align_corners=False
                ).squeeze(dim=1),
                target.squeeze(dim=1),
            )
        for opt in opts:
            opt.zero_grad()

        with amp.scale_loss(loss, opts) as scaled_loss:
            scaled_loss.backward()

        if grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        for opt in opts:
            opt.step()

        loss_meter.update(loss.item(), input.size(0))
        if idx>=3:
            batch_time.update(time() - end)
        end = time()

    return loss.item(), loss_meter.avg, batch_time.avg


def amp_validate(model, metrics, dataloader, stage=0, epoch=0):
    """Full Validation Pipeline.

    Support multiple metrics (but 1 per modality), multiple outputs.
    Assumes that the dataloader outputs have the correct type, that the model \
    outputs do not require any post-processing bar the upsampling \
    to the target size.
    Metrics and model's outputs must have the same length, and correspond to \
    the same keys as in the ordered dict of dataloader's sample.

    Args:
        model : PyTorch model object.
        metrics  : list of metric classes. Each metric class must have update
                   and val functions, and must have 'name' attribute.
        dataloader : iterable over samples.
                     Each sample must contain `image` key and
                     >= 1 optional keys.

    """
    device = next(model.parameters()).device
    model.eval()
    metrics = dt.misc.utils.make_list(metrics)
    for metric in metrics:
        metric.reset()

    pbar = dataloader

    def get_val(metrics):
        results = [(m.name, m.val()) for m in metrics]
        names, vals = list(zip(*results))
        out = ["{} : {:4f}".format(name, val) for name, val in results]
        return vals, " | ".join(out)

    with torch.no_grad():
        for idx, sample in enumerate(pbar):
            input, targets = get_input_and_targets(
                sample=sample, dataloader=dataloader, device=device
            )
            targets = [target.squeeze(dim=1).cpu().numpy() for target in targets]
            outputs = model(input)
            outputs = dt.misc.utils.make_list(outputs)
            for out, target, metric in zip(outputs, targets, metrics):
                metric.update(
                    F.interpolate(
                        out, size=target.shape[1:], mode="bilinear", align_corners=False
                    )
                    .squeeze(dim=1)
                    .cpu()
                    .numpy(),
                    target,
                )
            if idx%500==0:
                print("val", idx, get_val(metrics)[1])
        print(f"Validation: stage {stage} epoch {epoch}", get_val(metrics)[1])
    vals, _ = get_val(metrics)
    print("----" * 5)
    return vals

def amp_get_loaders(
    train_batch_size,
    val_batch_size,
    train_set,
    val_set,
    num_stages=1,
    num_workers=4,
    train_shuffle=True,
    val_shuffle=False,
    train_pin_memory=False,
    val_pin_memory=False,
    train_drop_last=False,
    val_drop_last=False,
    distributed=False
):
    """Create train and val loaders"""
    train_batch_sizes = dt.misc.utils.broadcast(train_batch_size, num_stages)
    train_sets = dt.misc.utils.broadcast(train_set, num_stages)
    if distributed:
        train_sampler = [torch.utils.data.distributed.DistributedSampler(train_sets[i]) for i in range(num_stages)]
    else:
        train_sampler = [None for i in range(num_stages)]
    train_loaders = [
        DataLoader(
            train_sets[i],
            batch_size=train_batch_sizes[i],
            shuffle=(train_sampler[i] is None),
            num_workers=num_workers,
            pin_memory=train_pin_memory,
            drop_last=train_drop_last,
            sampler=train_sampler[i]
        )
        for i in range(num_stages)
    ]
    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=val_shuffle,
        num_workers=num_workers,
        pin_memory=val_pin_memory,
        drop_last=val_drop_last,
    )
    return train_loaders, val_loader, train_sampler

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s",
        level=logging.INFO,
    )
    main()
