# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# =======================================================================

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
# ============================================================================

# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import time
from datetime import timedelta

import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.distributed as dist


from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter
from apex import amp, optimizers
#from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from vitutils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from vitutils.data_utils import get_loader
from vitutils.dist_util import get_world_size
import sys
import os

NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


logger = logging.getLogger(__name__)


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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    num_classes = 10 if args.dataset == "cifar10" else 100

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.npu.manual_seed_all(args.seed)


def valid(args, model,test_loader, global_step):
#def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        #if args.ddp:
            #if isinstance(test_loader, torch.utils.data.DataLoader):
                #test_loader.sampler.set_epoch(step)
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        # NLLLoss int64 is not supported,need change dtype
        y = y.to(torch.int32)
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    #writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy


def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        #writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    if args.npu_fused_sgd:
        optimizer = optimizers.NpuFusedSGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    print('lr',args.learning_rate)
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level,
                                          combine_grad=True)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20


    if args.ddp:
        model = model.to(f'npu:{NPU_CALCULATE_DEVICE}')
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[NPU_CALCULATE_DEVICE], broadcast_buffers=False)

    # Distributed training
    if args.local_rank != -1:
        #model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        fps = AverageMeter()
        end = time.time()
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        #添加prof图
        print('len',len(epoch_iterator))
        num_steps = 0
        for step, batch in enumerate(epoch_iterator):
            if num_steps > args.stop_step:
                if args.profiling == 'GE' or args.profiling == 'CANN':
                    import sys
                    sys.exit()
            elif num_steps <= args.stop_step and num_steps >= args.start_step  and args.profiling == 'CANN':
                with torch.npu.profile(profiler_result_path="./CANN_prof"):
                    if args.ddp:
                        if isinstance(train_loader, torch.utils.data.DataLoader):
                            train_loader.sampler.set_epoch(step)
                    batch = tuple(t.to(args.device) for t in batch)
                    x, y = batch
                    y = y.to(torch.int32)
                    loss = model(x, y)

                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        losses.update(loss.item() * args.gradient_accumulation_steps)
                        if args.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        scheduler.step()
                        optimizer.step()
                        optimizer.zero_grad()
                            #添加pr添加prof图

                        global_step += 1
                        fps.update(args.train_batch_size / (time.time() - end))
                        end = time.time()
            elif num_steps <= args.stop_step and num_steps >= args.start_step and args.profiling == 'GE':
                with torch.npu.profile(profiler_result_path="./GE_prof"):
                    if args.ddp:
                        if isinstance(train_loader, torch.utils.data.DataLoader):
                            train_loader.sampler.set_epoch(step)
                    batch = tuple(t.to(args.device) for t in batch)
                    x, y = batch
                    y = y.to(torch.int32)
                    loss = model(x, y)

                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        losses.update(loss.item() * args.gradient_accumulation_steps)
                        if args.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        scheduler.step()
                        optimizer.step()
                        optimizer.zero_grad()
                            #添加pr添加prof图

                        global_step += 1
                        fps.update(args.train_batch_size / (time.time() - end))
                        end = time.time()
            else:
                if args.ddp:
                    if isinstance(train_loader, torch.utils.data.DataLoader):
                        train_loader.sampler.set_epoch(step)
                            #if isinstance(test_loader, torch.utils.data.DataLoader):
                            #    test_loader.sampler.set_epoch(step)

                batch = tuple(t.to(args.device) for t in batch)
                x, y = batch
                y = y.to(torch.int32)
                loss = model(x, y)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    losses.update(loss.item() * args.gradient_accumulation_steps)
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()
                        #添加pr添加prof图

                    global_step += 1
                    fps.update(args.train_batch_size / (time.time() - end))
                    if global_step < 5 :
                        print("Iter_time: {:.4f}".format(time.time() - end))
                    end = time.time()

                    epoch_iterator.set_description(
                        "Training (%d / %d Steps) (loss=%2.5f) (FPS=%.2f)" % (global_step, t_total, losses.val,fps.val)
                    )
                    #if args.local_rank in [-1, 0]:
                       # writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                       # writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                    if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                        accuracy = valid(args, model, test_loader, global_step)
                        #accuracy = valid(args, model, writer, test_loader, global_step)
                        if best_acc < accuracy:
                        #save ckpt
                            save_model(args, model)
                            best_acc = accuracy
                        model.train()

                    if global_step % t_total == 0:
                        break
            num_steps = num_steps + 1


        losses.reset()
        if global_step % t_total == 0:
            break

    #if args.local_rank in [-1, 0]:
        #writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--addr', default='10.136.181.115', type=str, help='master addr')
    parser.add_argument("--data_dir", default=".", type=str,
                        help="path to dataset.")

    parser.add_argument('--npu-fused-sgd', action='store_true')
    parser.add_argument('--combine-grad', action='store_true')
    parser.add_argument( '--ddp',default=False,help='distributed or not ')
    parser.add_argument('--profiling', type=str, default='NONE',
                        help='choose profiling way--CANN,GE,NONE')
    parser.add_argument('--start_step', default=0, type=int,
                        help='start_step')
    parser.add_argument('--stop_step', default=1000, type=int,
                        help='stop_step')
    parser.add_argument('--bin', type=bool, default=False,
                        help='if bin')
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = args.addr  # '10.136.181.51'
    os.environ['MASTER_PORT'] = '29501'
    if args.bin:
        torch.npu.set_compile_mode(jit_compile=False)

    if args.ddp:
        NPU_WORLD_SIZE = int(os.getenv('NPU_WORLD_SIZE'))
        RANK = int(os.getenv('RANK'))
        torch.distributed.init_process_group('hccl', rank=RANK, world_size=NPU_WORLD_SIZE)

    print('lr',args.learning_rate)
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')
        device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')
        args.n_gpu = torch.npu.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.npu.set_device(args.local_rank)
        device = torch.device("npu", args.local_rank)
        torch.distributed.init_process_group(backend='hccl',
                                             world_size=int(os.environ["RANK_SIZE"]),
                                             rank=args.local_rank)
        args.n_gpu = 1

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)


if __name__ == "__main__":
    main()
