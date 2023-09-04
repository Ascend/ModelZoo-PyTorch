# -*- coding: utf-8 -*-
"""
Copyright 2023 Huawei Technologies Co., Ltd

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

import logging
import time
from contextlib import nullcontext

# if your python version < 3.7 use the below one
# from contextlib import suppress as nullcontext
import torch
from torch.nn.utils import clip_grad_norm_


class Executor:

    def __init__(self, global_start_time=0, **kwargs):
        self.step = 0
        self.total_time = 0.0
        self.total_train_time = 0.0
        self.total_eval_time = 0.0
        self.maximum_fps = 0.0
        self.e2e_train_data_num = 0
        self.global_start_time = global_start_time

    def train(self, model, optimizer, scheduler, data_loader, device, writer,
              args, scaler):
        ''' Train one epoch
        '''
        start_time = time.time()
        model.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        accum_grad = args.get('accum_grad', 1)
        is_distributed = args.get('is_distributed', True)
        use_amp = args.get('use_amp', False)
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(accum_grad))
        if use_amp:
            assert scaler is not None
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        model_context = nullcontext
        num_seen_utts = 0
        total_train_data_num = 0
        with model_context():
            for batch_idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, target_lengths = batch
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                target_lengths = target_lengths.to(device)
                num_utts = target_lengths.size(0)
                total_train_data_num += feats.shape[0]
                if num_utts == 0:
                    continue
                context = None
                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if is_distributed and batch_idx % accum_grad != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext
                with context():
                    # autocast context
                    # The more details about amp can be found in
                    # https://pytorch.org/docs/stable/notes/amp_examples.html
                    with torch.cuda.amp.autocast(scaler is not None):
                        loss_dict = model(feats, feats_lengths, target,
                                          target_lengths)
                        loss = loss_dict['loss'] / accum_grad
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                num_seen_utts += num_utts
                if batch_idx % accum_grad == 0:
                    if rank == 0 and writer is not None:
                        writer.add_scalar('train_loss', loss.item(), self.step)
                    # Use mixed precision training
                    if use_amp:
                        scaler.unscale_(optimizer)
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        # Must invoke scaler.update() if unscale_() is used in
                        # the iteration to avoid the following error:
                        #   RuntimeError: unscale_() has already been called
                        #   on this optimizer since the last update().
                        # We don't check grad here since that if the gradient
                        # has inf/nan values, scaler.step will skip
                        # optimizer.step().
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        if torch.isfinite(grad_norm):
                            optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    self.step += 1
                if batch_idx % log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                        epoch, batch_idx,
                        loss.item() * accum_grad)
                    for name, value in loss_dict.items():
                        if name != 'loss' and value is not None:
                            log_str += '{} {:.6f} '.format(name, value.item())
                    log_str += 'lr {:.8f} rank {}'.format(lr, rank)
                    logging.debug(log_str)
        end_time = time.time()
        train_time = end_time - start_time
        self.total_train_time += train_time
        fps = total_train_data_num / train_time
        self.maximum_fps = max(fps, self.maximum_fps)
        self.e2e_train_data_num = total_train_data_num
        total_data = total_train_data_num * (epoch + 1)
        train_avg_fps = total_data / self.total_train_time
        print('============================================================')
        print('device: {}, {}th epoch total training data num: {}'.format(rank, epoch + 1, total_train_data_num))
        print('device: {}, {}th epoch training time: {}'.format(rank, epoch + 1, train_time))
        print('device: {}, total training time: {}'.format(rank, self.total_train_time))
        print('device: {}, {}th epoch training fps: {}'.format(rank, epoch + 1, fps))
        print('device: {}, current maximum training fps: {}'.format(rank, self.maximum_fps))
        print('device: {}, training average fps: {}'.format(rank, train_avg_fps))
        print('============================================================')

    def cv(self, model, data_loader, device, args):
        ''' Cross validation on
        '''
        eval_start_time = time.time()
        model.eval()
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        log_interval = args.get('log_interval', 10)
        # in order to avoid division by 0
        num_seen_utts = 1
        total_loss = 0.0
        total_eval_data_num = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, target_lengths = batch
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                target_lengths = target_lengths.to(device)
                num_utts = target_lengths.size(0)
                total_eval_data_num += feats.shape[0]
                if num_utts == 0:
                    continue
                loss_dict = model(feats, feats_lengths, target, target_lengths)
                loss = loss_dict['loss']
                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                if batch_idx % log_interval == 0:
                    log_str = 'CV Batch {}/{} loss {:.6f} '.format(
                        epoch, batch_idx, loss.item())
                    for name, value in loss_dict.items():
                        if name != 'loss' and value is not None:
                            log_str += '{} {:.6f} '.format(name, value.item())
                    log_str += 'history loss {:.6f}'.format(total_loss /
                                                            num_seen_utts)
                    log_str += ' rank {}'.format(rank)
                    logging.debug(log_str)
        eval_end_time = time.time()
        eval_time = eval_end_time - eval_start_time
        print('> device: {}, {}th epoch evaluation time : {}'.format(rank, epoch + 1, eval_time))

        e2e_time = eval_end_time - self.global_start_time
        total_data = (self.e2e_train_data_num + total_eval_data_num) * (epoch + 1)
        e2e_fps = total_data / e2e_time
        print('============================================================')
        print('device: {}, {}th epoch total data num:{}'.format(rank, epoch + 1, total_data))
        print('device: {}, {}th epoch e2e time: {}'.format(rank, epoch + 1, e2e_time))
        print('device: {}, {}th epoch e2e fps: {}'.format(rank, epoch + 1, e2e_fps))
        print('============================================================')
        return total_loss, num_seen_utts
