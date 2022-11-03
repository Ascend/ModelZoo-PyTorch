# coding=utf-8
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

"""Sample Generate GPT2"""

import os
import random
import numpy as np
import torch
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.nn.functional as F
import argparse
import time
import json
import pickle
from arguments import get_args
from utils import Timers
from utils import load_checkpoint
from data_utils.tokenization_gpt2 import GPT2Tokenizer
from configure_data import configure_data
import mpu
import json

from tqdm import tqdm
from fp16 import FP16_Module
from model import GPT2Model
from model import DistributedDataParallel as DDP
from utils import print_rank_0
from data.samplers import DistributedBatchSampler, RandomSampler

from torch.utils.data import TensorDataset

from pretrain_gpt2 import *

def yprint(str):
    print("\033[43;30m{}\033[0m".format(str))

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.start_count_index = 10

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        if self.count > (self.start_count_index * n):
            self.sum += val * n
            self.avg = self.sum / (self.count - self.start_count_index * n)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("[npu id:", '0', "]", '\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class CHIDDataset(torch.utils.data.Dataset):
    def __init__(self, args, data_path, split, tokenizer, ratio=1):
        self.split = split
        self.tokenizer = tokenizer
        self.ratio = ratio
        self.args = args
        self.world_size = args.world_size

        self.pad_id = tokenizer.encoder['<pad>']
        self.eod_token = tokenizer.encoder['<eod>']
        args.eod_token = tokenizer.encoder['<eod>']

        with open(data_path, "r") as f:
            data = json.load(f)

        self.samples, self.sizes, self.truth_labels = self.process(data)
        if max(self.sizes) % 16 != 0:
            self.max_size = (max(self.sizes) // 16 + 1) * 16
        else:
            self.max_size = max(self.sizes)

    def process(self, data):
        contents = data["contents"]
        sids = data["sids"]
        truth_labels = data["labels"]
        cids = data["cids"]
        sizes = []
        samples = []
        for content, sid, cid in zip(tqdm(contents[:int(self.ratio * len(contents))], desc="Processing", disable=(torch.distributed.get_rank() != 0)), sids, cids):
            input_ids = content
            input_ids = input_ids + [self.eod_token]
            length = len(input_ids) - 1
            sizes.append(length)
            samples.append({
                "sid": sid,
                "cid": cid,
                "input_ids": input_ids[:-1],
                "loss_mask": [1.0] * length,
                "labels": input_ids[1:]
            })

        return samples, sizes, truth_labels

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, idx):
        return self.samples[idx], self.sizes[idx]

    def collate(self, x):
        bs = len(x)
        samps = [s[0] for s in x]
        sizes = [s[1] for s in x]
        max_size = self.max_size

        attn_mask = torch.tril(torch.ones((max_size, max_size))).unsqueeze(0)
        position_ids = torch.arange(max_size, dtype=torch.long).unsqueeze(0).repeat(bs, 1)
        if self.args.fp16:
            attn_mask = attn_mask.half()

        batch_sample = {
            "input_ids": torch.ones(bs, max_size).long() * self.pad_id,
            "attention_mask": attn_mask,
            "position_ids": position_ids
        }

        no_model_sample = {
            "sids": torch.zeros(bs).long(),
            "cids": torch.zeros(bs).long(),
            "loss_mask": torch.zeros(bs, max_size).float(),
            "labels": torch.ones(bs, max_size).long() * self.pad_id,
        }

        for i, samp in enumerate(samps):
            batch_sample["input_ids"][i, :len(samp["input_ids"])] = torch.tensor(samp["input_ids"])
            no_model_sample["loss_mask"][i, :len(samp["loss_mask"])] = torch.tensor(samp["loss_mask"])
            no_model_sample["labels"][i, :len(samp["labels"])] = torch.tensor(samp["labels"])
            no_model_sample["sids"][i] = torch.tensor(samp["sid"])
            no_model_sample["cids"][i] = torch.tensor(samp["cid"])

        return batch_sample, no_model_sample


def load_data(args, data_type, tokenizer, ratio=1):
    data_path = args.data_dir
    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = args.batch_size * world_size
    num_workers = args.num_workers

    # Dataset
    filename = os.path.join(data_path, data_type+'.json')
    dataset = CHIDDataset(args, filename, data_type, tokenizer, ratio=ratio)
    
    # Use a random sampler with distributed batch sampler.
    if data_type == 'train':
        sampler = RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(sampler=sampler,
                                            batch_size=global_batch_size,
                                            drop_last=True,
                                            rank=rank,
                                            world_size=world_size)
    
    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True,
                                       collate_fn=dataset.collate), dataset


def main():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # get the tokenizer
    tokenizer = GPT2Tokenizer(os.path.join(args.tokenizer_path, 'vocab.json'), os.path.join(args.tokenizer_path, 'chinese_vocab.model'))

    # load data
    test_dataloader, test_dataset = load_data(args, 'test', tokenizer, 1)
    # Set an arbitrary positive integer since the optimizer and the scheduler will not be used when do eval.
    args.train_iters = 1

    # Model
    model, _, _ = setup_model_and_optimizer(args)

    device = torch.npu.current_device()

    # give a time stemp to the model
    cur_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    results_dir = os.path.join(args.results_dir, "{}-{}".format(args.model_name, cur_time))

    if torch.distributed.get_rank() == 0:
        os.makedirs(results_dir, exist_ok=True)

    model.eval()
    all_sids = []
    all_cids = []
    all_losses = []
    with torch.no_grad():
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses_show = AverageMeter('Loss', ':.4e')
        fps = AverageMeter('it/s', ':.4e')
        progress = ProgressMeter(
            len(test_dataloader),
            [batch_time, data_time, losses_show, fps],
            prefix="Test: ")
        end = time.time()
        i = 0
        for batch, no_model_batch in test_dataloader:
            data_time.update(time.time() - end)
            for k in batch:
                batch[k] = batch[k].to(device)
            for k in no_model_batch:
                no_model_batch[k] = no_model_batch[k].to(device)

            output = model(**batch)
            losses = mpu.vocab_parallel_cross_entropy(output.contiguous().float(), no_model_batch["labels"], args)
            loss_mask = no_model_batch["loss_mask"]
            loss = torch.sum(losses * loss_mask, dim=-1) / loss_mask.sum(dim=-1)

            if loss != None:
                losses_show.update(loss.item())
            if i >= 10:
                t = time.time()
                batch_time.update(t - end)
                fps.update(args.batch_size / batch_time.val)
                progress.display(i)
            end = time.time()
            i = i + 1
            loss_tensor_list = [torch.zeros_like(loss).to(device) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(loss_tensor_list, loss.data, group=mpu.get_data_parallel_group())
            for loss_item in loss_tensor_list:
                all_losses.append(loss_item.cpu())

            sids = no_model_batch["sids"]
            # HCCL不支持int64
            sids = sids.int()
            sid_tensor_list = [torch.zeros_like(sids) for _ in range(mpu.get_data_parallel_world_size())]
            if args.model_parallel_size > 1:
                torch.distributed.all_gather(sid_tensor_list, sids.data, group=mpu.get_data_parallel_group())
                for sid_item in sid_tensor_list:
                    all_sids.append(sid_item.cpu())
            else:
                all_sids.extend(sids.cpu().data)

            cids = no_model_batch["cids"]
            cids = cids.int()
            cid_tensor_list = [torch.zeros_like(cids) for _ in range(mpu.get_data_parallel_world_size())]
            if args.model_parallel_size > 1:
                torch.distributed.all_gather(cid_tensor_list, cids.data, group=mpu.get_data_parallel_group())
                for cid_item in cid_tensor_list:
                    all_cids.append(cid_item.cpu())
            else:
                all_cids.extend(cids.cpu().data)

    if torch.distributed.get_rank() == 0:
        all_losses = torch.stack(all_losses).view(-1).cpu().detach().numpy()
        all_sids = torch.stack(all_sids).view(-1).cpu().detach().numpy()
        all_cids = torch.stack(all_cids).view(-1).cpu().detach().numpy()

        truth_labels = test_dataset.truth_labels
        preds = [[] for _ in truth_labels]

        for sid, cid, loss in zip(all_sids, all_cids, all_losses):
            preds[sid].append((cid, loss))

        preds = [min(p, key=lambda x: x[1])[0] for p in preds if len(p) > 0]

        yprint("Acc: {}".format(sum([int(p == l) for p, l in zip(preds, truth_labels)]) / len(truth_labels)))
        with open(os.path.join(results_dir, "zero-shot_result.txt"), "w") as f:
            f.write("Acc: {}\n".format(sum([int(p == l) for p, l in zip(preds, truth_labels)]) / len(truth_labels)))

    if args.model_parallel_size > 1:
        torch.distributed.barrier()

if __name__ == "__main__":

    main()
