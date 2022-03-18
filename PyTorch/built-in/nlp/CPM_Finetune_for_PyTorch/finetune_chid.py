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
import torch.nn.functional as F
import argparse
import time
import json
import pickle
from tqdm import tqdm
from arguments import get_args
from utils import Timers
from pretrain_gpt2 import initialize_distributed
from pretrain_gpt2 import set_random_seed
from pretrain_gpt2 import get_train_val_test_data
from pretrain_gpt2 import get_masks_and_position_ids
from utils import load_checkpoint
# from data_utils import make_tokenizer
from data_utils.tokenization_gpt2 import GPT2Tokenizer
from configure_data import configure_data
import mpu
import deepspeed
import json
import time

from tqdm import tqdm
from fp16 import FP16_Module
from model import GPT2Model
from model import DistributedDataParallel as DDP
from utils import print_rank_0
from data.samplers import DistributedBatchSampler, RandomSampler
# from sklearn.metrics import accuracy_score

from torch.utils.data import TensorDataset

from pretrain_gpt2 import *

def yprint(str):
    print("\033[43;30m{}\033[0m".format(str))

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
            # cand_ids: the candidate label ids, namely, ids of "0", "1", ..., "9"
            # data: preprocessed (tokenized) data
            self.cand_ids, data = json.load(f)
        self.samples, self.sizes = self.process(data)

        self.max_size = max(self.sizes)

    def process(self, data):
        samples, sizes = [], []
        for d in tqdm(data[:int(self.ratio * len(data))], disable=(torch.distributed.get_rank() != 0)):
            # only use the loss of the last token
            loss_mask = [0] * (len(d["sent"]) - 2) + [1]

            samples.append({
                "input_ids": d["sent"][:-1], # ids for the tokenized sentence
                "loss_mask": loss_mask, # mask of the loss
                "labels": d["sent"][1:], # token labels of each sentence
                "truth": d["truth"], # labels if each sentence, should be an integer in [0, 9]
            })
            sizes.append(len(d["sent"]) - 1)

        return samples, sizes

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, idx):
        return self.samples[idx], self.sizes[idx]

    def collate(self, x):
        bs = len(x)
        samps = [s[0] for s in x]
        sizes = [s[1] for s in x]

        # fit to the max_size
        max_size = self.max_size

        # triangle attention mask
        attn_mask = torch.tril(torch.ones((max_size, max_size))).unsqueeze(0)
        position_ids = torch.arange(max_size, dtype=torch.long).unsqueeze(0).repeat(bs, 1)

        if self.args.fp16:
            attn_mask = attn_mask.half()

        # the data that need to go through the model
        batch_sample = {
            "input_ids": torch.ones(bs, max_size).long() * self.pad_id,
            "attention_mask": attn_mask.unsqueeze(1),
            "position_ids": position_ids,
        }

        # the data that do not need to go through the model
        no_model_sample = {
            "labels": torch.ones(bs, max_size).long() * self.pad_id,
            "truth": torch.zeros(bs).long(),
            "loss_mask": torch.zeros(bs, max_size).float()
        }

        for i, samp in enumerate(samps):
            batch_sample["input_ids"][i, :len(samp["input_ids"])] = torch.tensor(samp["input_ids"])
            no_model_sample["labels"][i, :len(samp["labels"])] = torch.tensor(samp["labels"])
            no_model_sample["truth"][i] = torch.tensor(samp["truth"])
            no_model_sample["loss_mask"][i, :len(samp["loss_mask"])] = torch.tensor(samp["loss_mask"])

        return batch_sample, no_model_sample

def load_data(args, data_type, tokenizer, ratio=1):
    data_path = args.data_dir

    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = args.batch_size * world_size
    num_workers = args.num_workers

    # Dataset
    filename = os.path.join(data_path, data_type + '.json')
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


def evaluate(args, model, dataloader, cand_ids, device, mode="dev"):
    model.eval()
    all_truth, all_preds = [], []
    with torch.no_grad():
        for batch, no_model_batch in tqdm(dataloader, desc="Evaluating {}".format(mode), disable=(torch.distributed.get_rank() != 0)):
            for k in batch:
                batch[k] = batch[k].to(device)
            for k in no_model_batch:
                no_model_batch[k] = no_model_batch[k].to(device)

            output = model(**batch)
            output = torch.sum(output * no_model_batch["loss_mask"].unsqueeze(-1), 1) / torch.sum(no_model_batch["loss_mask"], -1).unsqueeze(-1)

            # gather the output logits from other gpus
            tensor_list = [torch.zeros_like(output) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(tensor_list, output, mpu.get_data_parallel_group())

            # gather the truth labels from other gpus
            tensor_list_truth = [torch.zeros_like(no_model_batch["truth"], dtype=torch.long) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(tensor_list_truth, no_model_batch["truth"], mpu.get_data_parallel_group())

            if args.model_parallel_size == 1:
                scores = torch.stack(tensor_list, 0).view(-1, 30000)
            else:
                assert args.model_parallel_size == 2, "Now, we only support model parallel <= 2"
                # for convience implementation. Note that the truth labels only appears in the first 15000 part of the logits, e.g. on rank 0, 2, 4, ...
                scores = torch.stack(tensor_list, 0).view(-1, 15000)

            truth = torch.stack(tensor_list_truth, 0)
            truth = truth.view(-1)
            scores = scores[:, cand_ids]

            preds = torch.argmax(scores, dim=-1)

            all_truth.extend(truth.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())
        
    acc = sum([int(p == l) for p, l in zip(all_preds, all_truth)]) / len(all_truth)
    acc = torch.tensor(acc).to(device)
    
    acc_list = [torch.zeros_like(acc) for _ in range(mpu.get_model_parallel_world_size())]
    torch.distributed.all_gather(acc_list, acc, mpu.get_model_parallel_group())

    return acc_list[0].item(), all_truth, all_preds


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

    # load train data
    if args.do_train:
        train_dataloader, _ = load_data(args, 'train', tokenizer, 1)
        dev_dataloader, dev_dataset = load_data(args, 'dev', tokenizer, 1)

        with open(args.deepspeed_config, "r") as f:
            deepspeed_conf = json.load(f)

        epoch = args.epoch
        grad_acc = deepspeed_conf["gradient_accumulation_steps"]
        args.train_iters = len(train_dataloader) * epoch / grad_acc

        # Model, optimizer, and learning rate.
        # TODO: maybe need to reinitialize optimizer
    elif args.do_eval:
        # Set an arbitrary positive integer since the optimizer and the scheduler will not be used when do eval.
        args.train_iters = 1

    model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    device = torch.cuda.current_device()

    # give a time stemp to the model
    cur_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    results_dir = os.path.join(args.results_dir, "{}-{}".format(args.model_name, cur_time))
    os.makedirs(results_dir, exist_ok=True)

    if args.do_train and torch.distributed.get_rank() == 0:

        with open(os.path.join(results_dir, "train_log.txt"), "w") as f:
            f.write("Train losses:\n")

        with open(os.path.join(results_dir, "dev_log.txt"), "w") as f:
            f.write("Dev accs:\n")

    torch.distributed.barrier()

    if args.do_train:
        cand_ids = torch.tensor(dev_dataset.cand_ids).to(device)
        total_loss, logging_loss, best_acc = 0.0, 0.0, 0.0
        global_step, total_step, best_step = 0, 0, 0
        
        for e in range(epoch):
            model.train()
            for batch, no_model_batch in tqdm(train_dataloader, disable=(torch.distributed.get_rank() != 0)):
                for k in batch:
                    batch[k] = batch[k].to(device)
                for k in no_model_batch:
                    no_model_batch[k] = no_model_batch[k].to(device)

                output = model(**batch)
                # get the loss of the last token
                output = torch.sum(output * no_model_batch["loss_mask"].unsqueeze(-1), 1) / torch.sum(no_model_batch["loss_mask"], -1).unsqueeze(-1)
                # get the label of the last token
                labels = no_model_batch["labels"].float()
                labels = (torch.sum(labels * no_model_batch["loss_mask"], 1) / torch.sum(no_model_batch["loss_mask"], -1)).long()
                # cross_entropy loss
                losses = mpu.vocab_parallel_cross_entropy(output.unsqueeze(1).contiguous().float(), labels.unsqueeze(1))
                loss = torch.mean(losses)

                model.backward(loss)
                model.step()

                torch.distributed.all_reduce(loss.data, group=mpu.get_data_parallel_group())
                loss.data = loss.data / mpu.get_data_parallel_world_size()
                total_loss += loss.item() / grad_acc

                if total_step % grad_acc == 0:
                    global_step += 1
                    if global_step != 0 and global_step % args.log_interval == 0:
                        # logging
                        if torch.distributed.get_rank() == 0:
                            train_log = "Epoch {}, global step {}, total step {}, train lm loss: {}".format(e, global_step, epoch * len(train_dataloader), (total_loss - logging_loss) / args.log_interval)
                            yprint(train_log)
                            with open(os.path.join(results_dir, "train_log.txt"), "a") as f:
                                f.write(train_log + "\n")

                        logging_loss = total_loss
    
                    if global_step != 0 and global_step % args.eval_interval == 0:
                        # evaluate on the dev
                        acc, _, _ = evaluate(args, model, dev_dataloader, cand_ids, device, mode="dev")
                        dev_results_dir = os.path.join(results_dir, "dev_step-{}".format(global_step))

                        if acc > best_acc:
                            best_acc = acc
                            best_step = global_step

                        if torch.distributed.get_rank() == 0:
                            # we will only write the log file once
                            dev_log = "Epoch: {}, Global step: {}, Acc: {}".format(e, global_step, acc)
                            yprint(dev_log)
                            os.makedirs(dev_results_dir, exist_ok=True)
                            with open(os.path.join(dev_results_dir, "dev_result.txt"), "w") as f:
                                f.write(dev_log + "\n")
                            with open(os.path.join(results_dir, "dev_log.txt"), "a") as f:
                                f.write(dev_log + "\n")

                        torch.distributed.barrier()
                        
                        args.save = dev_results_dir
                        save_checkpoint(global_step, model, optimizer, lr_scheduler, args)

                total_step += 1

        with open(os.path.join(dev_results_dir, "dev_log.txt"), "a") as f:
            f.write("Best acc: {} Best step: {}\n".format(best_acc, best_step))

    if args.do_eval:
        # evaluate on the test
        test_dataloader, test_dataset = load_data(args, 'test', tokenizer, 1)
        cand_ids = torch.tensor(test_dataset.cand_ids).to(device)

        if args.do_train:
            # if do training, then evaluate the one with the max acc on dev set.
            eval_ckpt_path = os.path.join(results_dir, "dev_step-{}".format(best_step))
            args.load = eval_ckpt_path
        else:
            # if only do eval, then evaluate the one specified by the user.
            args.load = args.eval_ckpt_path            
        
        load_checkpoint(model=model, optimizer=None, lr_scheduler=None, args=args)
        acc, _, _ = evaluate(args, model, test_dataloader, cand_ids, device, mode="test")

        if torch.distributed.get_rank() == 0:
            eval_log = "Checkpoint from {}: Acc: {}".format(args.load, acc)
            yprint(eval_log)
            with open(os.path.join(results_dir, "eval_log"), "w") as f:
                f.write(eval_log + "\n")

        torch.distributed.barrier()


if __name__ == "__main__":

    main()
