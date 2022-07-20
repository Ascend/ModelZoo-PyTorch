
#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from multiprocessing import cpu_count
import numpy as np
import argparse
import sys
import os
import time
import math

from model import FastSpeech
from loss import DNNLoss
from dataset import BufferDataset, DataLoader
from dataset import get_data_to_buffer, collate_fn_tensor
import hparams as hp
import utils
import torch.npu
from apex import amp
import apex

NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

# 添加二进制模糊编译机制
torch.npu.set_compile_mode(jit_compile=False)
# 添加算子黑名单
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "DynamicGRUV2,BroadcastTo,BNTrainingUpdateGrad,Slice,MatMul,Cast"
torch.npu.set_option(option)


def step_and_update_lr_frozen(optimizer, learning_rate_frozen):
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate_frozen


def get_learning_rate(optimizer):
    learning_rate = 0.0
    for param_group in optimizer.param_groups:
        learning_rate = param_group['lr']
    return learning_rate


def get_lr_scale(current_steps,warmup_steps):
    return np.min([
        np.power(current_steps, -0.5),
        np.power(warmup_steps, -1.5) * current_steps])


def update_learning_rate(optimizer,init_lr,current_steps,warmup_steps):
    ''' Learning rate scheduling per step '''
    lr = init_lr * get_lr_scale(current_steps,warmup_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main(args):


    distributed = int(os.environ['RANK_SIZE']) > 1
    if distributed:
        world_size = int(os.environ['RANK_SIZE'])
        rank_id = int(os.environ['RANK_ID'])
        torch.distributed.init_process_group("hccl", rank=rank_id, world_size=world_size)


    # Define model
    print("Use FastSpeech",flush=True)
    model = FastSpeech().to(f'npu:{NPU_CALCULATE_DEVICE}')
    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of TTS Parameters:', num_param)
    # Get buffer
    print("Load data to buffer")
    buffer = get_data_to_buffer()

    # Optimizer and loss
    if args.use_npu_fused_adam:
        print("Starting to use npu fused adam optimizer !!!",flush=True)
        optimizer = apex.optimizers.NpuFusedAdam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)

    if args.apex:
        print("Starting to use apm.initialize !!!",flush=True)
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level,
                                          scale_window=20,
                                          combine_grad=True)
    init_lr = np.power(hp.decoder_dim, -0.5)
    fastspeech_loss = DNNLoss().to(f'npu:{NPU_CALCULATE_DEVICE}',non_blocking=True)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[NPU_CALCULATE_DEVICE], find_unused_parameters=True)
    # Load checkpoint if exists
    try:
        checkpoint = torch.load(os.path.join(
            hp.checkpoint_path, 'checkpoint_%d.pth.tar' % args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step %d---\n" % args.restore_step)
    except:
        print("\n---Start New Training---\n",flush=True)
        if not os.path.exists(hp.checkpoint_path):
            os.mkdir(hp.checkpoint_path)

    # # Init logger
    # if not os.path.exists(hp.logger_path):
    #     os.mkdir(hp.logger_path)

    # Get dataset
    dataset = BufferDataset(buffer)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if distributed else None
    # Get Training Loader
    training_loader = DataLoader(dataset,
                                 batch_size=hp.batch_expand_size * hp.batch_size,
                                 shuffle=(train_sampler is None),
                                 collate_fn=collate_fn_tensor,
                                 sampler=train_sampler,
                                 drop_last=True,
                                 num_workers=0)
    total_step = hp.epochs * len(training_loader) * hp.batch_expand_size
    print("++++++batch_size+++++++++:%s"%(hp.batch_expand_size * hp.batch_size),flush=True)
    # Define Some Information
    Time = np.array([])
    Start = time.perf_counter()

    # Training
    model.train()

    for epoch in range(hp.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        for i, batchs in enumerate(training_loader):
            # real batch start here
            if args.perf and i >= 1:
                break 
            for j, db in enumerate(batchs):
                if args.perf and  j >= 5:
                    break
                start = time.time()
                start_time = time.perf_counter()

                current_step = i * hp.batch_expand_size + j + args.restore_step + \
                    epoch * len(training_loader) * hp.batch_expand_size + 1

                # Init
                optimizer.zero_grad()

                # Get Data
                #character = db["text"].long().to(device)
                #mel_target = db["mel_target"].float().to(device)
                #duration = db["duration"].int().to(device)
                #mel_pos = db["mel_pos"].long().to(device)
                #src_pos = db["src_pos"].long().to(device)
                character = db["text"].int().to(f'npu:{NPU_CALCULATE_DEVICE}',non_blocking=True)
                mel_target = db["mel_target"].float().to(f'npu:{NPU_CALCULATE_DEVICE}',non_blocking=True)
                duration = db["duration"].int().to(f'npu:{NPU_CALCULATE_DEVICE}',non_blocking=True)
                mel_pos = db["mel_pos"].int().to(f'npu:{NPU_CALCULATE_DEVICE}',non_blocking=True)
                src_pos = db["src_pos"].int().to(f'npu:{NPU_CALCULATE_DEVICE}',non_blocking=True)
                max_mel_len = db["mel_max_len"]
                # Forward
                mel_output, mel_postnet_output, duration_predictor_output = model(character,src_pos,mel_pos=mel_pos,mel_max_length=max_mel_len,length_target=duration)

                # Cal Loss
                mel_loss, mel_postnet_loss, duration_loss = fastspeech_loss(mel_output,
                                                                            mel_postnet_output,
                                                                            duration_predictor_output,
                                                                            mel_target,
                                                                            duration)
                total_loss = mel_loss + mel_postnet_loss + duration_loss

                # Logger
                t_l = total_loss.item()
                m_l = mel_loss
                m_p_l = mel_postnet_loss
                d_l = duration_loss

                # with open(os.path.join("logger", "total_loss.txt"), "a") as f_total_loss:
                #     f_total_loss.write(str(t_l)+"\n")
                #
                # with open(os.path.join("logger", "mel_loss.txt"), "a") as f_mel_loss:
                #     f_mel_loss.write(str(m_l)+"\n")
                #
                # with open(os.path.join("logger", "mel_postnet_loss.txt"), "a") as f_mel_postnet_loss:
                #     f_mel_postnet_loss.write(str(m_p_l)+"\n")
                #
                # with open(os.path.join("logger", "duration_loss.txt"), "a") as f_d_loss:
                #     f_d_loss.write(str(d_l)+"\n")
                # Backward
                # compute gradient and do SGD step
                if args.apex:
                    with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_loss.backward()
                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(
                    model.parameters(), hp.grad_clip_thresh)

                # Update weights
                if args.frozen_learning_rate:
                    step_and_update_lr_frozen(optimizer,
                        args.learning_rate_frozen)
                else:
                    update_learning_rate(optimizer,init_lr,current_step,hp.n_warm_up_step)

                optimizer.step()

                # Print
                step_time = time.time() - start
                FPS = (hp.batch_expand_size * hp.batch_size if not distributed else hp.batch_expand_size * hp.batch_size * world_size) / step_time
                if current_step % hp.log_step == 0:
                    Now = time.perf_counter()

                    str1 = "Epoch [{}/{}], Step [{}/{}], time/step(s):{:.4f}, FPS:{:.3f}".format(
                        epoch+1, hp.epochs, current_step, total_step,step_time,FPS)
                    str2 = "Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f}, total_loss: {:.4f};".format(
                        m_l, m_p_l, d_l,t_l)
                    str3 = "Current Learning Rate is {:.6f}.".format(
                        get_learning_rate(optimizer))
                    str4 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                        (Now-Start), (total_step-current_step)*np.mean(Time))

                    print("\n" + str1,flush=True)
                    print(str2)
                    print(str3)
                    print(str4,flush=True)

                    # with open(os.path.join("logger", "logger.txt"), "a") as f_logger:
                    #     f_logger.write(str1 + "\n")
                    #     f_logger.write(str2 + "\n")
                    #     f_logger.write(str3 + "\n")
                    #     f_logger.write(str4 + "\n")
                    #     f_logger.write("\n")

                # if current_step % hp.save_step == 0:
                #     torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                #     )}, os.path.join(hp.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                #     print("save model at step %d ..." % current_step)

                end_time = time.perf_counter()
                Time = np.append(Time, end_time - start_time)
                if len(Time) == hp.clear_Time:
                    temp_value = np.mean(Time)
                    Time = np.delete(
                        Time, [i for i in range(len(Time))], axis=None)
                    Time = np.append(Time, temp_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    parser.add_argument('--frozen_learning_rate', type=bool, default=False)
    parser.add_argument("--learning_rate_frozen", type=float, default=1e-3)


    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1、O2 for mixed precision training.')
    parser.add_argument('--use-npu-fused-adam', action='store_true',
                        help='Use npu fused adam optimizer')
    parser.add_argument('--loss-scale-value', default=1., type=float,
                        help='loss scale using in amp, default -1 means dynamic')
    
    parser.add_argument('--perf', action='store_true',
                        help='run shell script about performance')
    args = parser.parse_args()

    main(args)

