# Copyright 2022 Huawei Technologies Co., Ltd
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

import argparse
import logging
import os
import time

import numpy as np
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import apex
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from apex import amp
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
from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolyScheduler
from partial_fc import PartialFC, PartialFCAdamW
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_distributed_sampler import setup_seed


try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group(backend="hccl", rank=rank, world_size=world_size)
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="hccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main(args):

    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)
    if not cfg.fp16:
        option = {}
        option["ACL_PRECISION_MODE"] = "must_keep_origin_dtype"
        torch.npu.set_option(option)
        torch.npu.config.allow_internal_format=False
    torch.npu.set_device(args.local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )

    train_loader = get_dataloader(
        cfg.rec,
        args.local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.seed,
        cfg.num_workers
    )

    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).npu()

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().npu()
        # TODO the params of partial fc must be last in the params list
        if cfg.fp16:
            opt_backbone = apex.optimizers.NpuFusedSGD(
                params=[{"params": backbone.parameters()}],
                lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
        else:
            opt_backbone = torch.optim.SGD(
                params=[{"params": backbone.parameters()}],
                lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

        opt_pfc = torch.optim.SGD(
            params=[{"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFCAdamW(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().npu()
        if cfg.fp16:
            opt_backbone = apex.optimizers.NpuFusedAdamW(
                params=[{"params": backbone.parameters()}],
                lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            opt_backbone = torch.optim.AdamW(
                params=[{"params": backbone.parameters()}],
                lr=cfg.lr, weight_decay=cfg.weight_decay)

        opt_pfc = torch.optim.AdamW(
            params=[{"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler_backbone = PolyScheduler(
        optimizer=opt_backbone,
        base_lr=cfg.lr,
        max_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step,
        last_epoch=-1
    )

    lr_scheduler_pfc = PolyScheduler(
        optimizer=opt_pfc,
        base_lr=cfg.lr,
        max_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step,
        last_epoch=-1
    )

    combine_ddp = True

    if cfg.fp16:
        backbone, [opt_backbone, opt_pfc] = amp.initialize(backbone, [opt_backbone, opt_pfc], opt_level="O1",
                                                           scale_window=100, combine_grad=True, combine_ddp=combine_ddp,
                                                           ddp_replica_count=6, loss_scale=256.)
    else:
        backbone, [opt_backbone, opt_pfc] = amp.initialize(backbone, [opt_backbone, opt_pfc], opt_level="O0",
                                                            loss_scale=1.)
    opt_pfc.accelerate = False
    opt_pfc.combine_ddp = False

    if combine_ddp:
        module_backbone = backbone
    else:
        backbone = torch.nn.parallel.DistributedDataParallel(
            module=backbone, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
            find_unused_parameters=True)
        module_backbone = backbone.module
    backbone.train()

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_npu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        module_backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt_backbone.load_state_dict(dict_checkpoint["state_optimizer_backbone"])
        opt_pfc.load_state_dict(dict_checkpoint["state_optimizer_pfc"])
        lr_scheduler_backbone.load_state_dict(dict_checkpoint["state_lr_scheduler_backbone"])
        lr_scheduler_pfc.load_state_dict(dict_checkpoint["state_lr_scheduler_pfc"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, summary_writer=summary_writer
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()

    for epoch in range(start_epoch, cfg.num_epoch):
        torch.distributed.barrier()
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        profiler = Profile(start_step=int(os.getenv("PROFILE_START_STEP", 10)),
                           profile_type=os.getenv("PROFILE_TYPE"))
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            # control exec time to avoid taking too long
            if args.perf_steps and global_step > args.perf_steps:
                exit(0)
            start_time = time.time()
            img = img.npu()
            local_labels = local_labels.npu()
            profiler.start()
            local_embeddings = backbone(img)
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels, opt_pfc)
            if cfg.fp16:
                with amp.scale_loss(loss, [opt_backbone, opt_pfc]) as scaled_loss:
                    scaled_loss.backward()
                opt_backbone.clip_optimizer_grad_norm_fused(5)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
            opt_backbone.step()
            opt_pfc.step()
            opt_backbone.zero_grad()
            opt_pfc.zero_grad()
            lr_scheduler_backbone.step()
            lr_scheduler_pfc.step()
            profiler.end()
            if global_step < 3 and epoch == 0:
                print("step_time = {}".format(time.time() - start_time), flush=True)
            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler_backbone.get_last_lr()[0], apex.amp._amp_state.loss_scalers[0])
                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)
                    torch.distributed.barrier()

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": module_backbone.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer_backbone": opt_backbone.state_dict(),
                "state_optimizer_pfc": opt_pfc.state_dict(),
                "state_lr_scheduler_backbone": lr_scheduler_backbone.state_dict(),
                "state_lr_scheduler_pfc": lr_scheduler_pfc.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_npu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(module_backbone.state_dict(), path_module)

        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(module_backbone.state_dict(), path_module)

        from torch2onnx import convert_onnx
        convert_onnx(module_backbone.cpu().eval(), path_module, os.path.join(cfg.output, "model.onnx"))
class NoProfiling():
    def __enter__(self):
        ...
    def __exit__(self, exc_type, exc_val, exc_tb):
        ...



if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    parser.add_argument("--perf_steps", type=int, default=0, help="number of steps on performance mode")
    parser.add_argument('--profiling', type=str, default='False',help='profiling')
    parser.add_argument('--start_step', default=90, type=int, help='start_step')
    parser.add_argument('--stop_step', default=100, type=int,help='stop_step')
    main(parser.parse_args())
