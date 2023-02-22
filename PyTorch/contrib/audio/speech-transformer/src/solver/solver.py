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

import os
import time
import torch
if torch.__version__ >= "1.8":
    import torch_npu
from transformer import Transformer
from loss import cal_performance
from utils import IGNORE_ID
from apex import amp

class Solver(object):
    """
    """

    def __init__(self, data, model, optimizer, args, package):
        self.is_distributed = args.is_distributed
        self.args = args
        self.package = package
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.optimizer = optimizer
        self.num_of_gpus = args.num_of_gpus
        # Low frame rate feature
        self.LFR_m = args.LFR_m
        self.LFR_n = args.LFR_n

        # Training config
        self.epochs = args.epochs
        self.label_smoothing = args.label_smoothing
        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.model_path = args.model_path
        # logging
        self.print_freq = args.print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self.visdom = args.visdom
        self.visdom_lr = args.visdom_lr
        self.visdom_epoch = args.visdom_epoch
        self.visdom_id = args.visdom_id
        if self.visdom:
            from visdom import Visdom
            self.vis = Visdom(env=self.visdom_id)
            self.vis_opts = dict(title=self.visdom_id,
                                 ylabel='Loss', xlabel='Epoch',
                                 legend=['train loss', 'cv loss'])
            self.vis_window = None
            self.vis_epochs = torch.arange(1, self.epochs + 1)
            self.optimizer.set_visdom(self.visdom_lr, self.vis)

        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            self.start_epoch = int(self.package.get('epoch', 1))
            self.tr_loss[:self.start_epoch] = self.package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = self.package['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False

    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            if self.is_distributed:
                self.args.tr_sampler.set_epoch(epoch) 
            print("Training...")
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)

            # Save model each epoch
            if self.checkpoint and epoch % 10 == 0 and self.args.local_rank == 0:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(Transformer.serialize(self.model,self.optimizer,epoch+1,self.LFR_m, self.LFR_n,self.package, tr_loss=self.tr_loss,cv_loss=self.cv_loss), file_path)
                print('Saving checkpoint model to %s' % file_path)

            # Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            val_loss = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f} | local_rank {3}'.format(
                      epoch + 1, time.time() - start, val_loss, self.args.local_rank))
            print('-' * 85)

            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.save_folder, self.model_path)
                if self.args.local_rank == 0:
                    torch.save(Transformer.serialize(self.model,self.optimizer,epoch+1,self.LFR_m, self.LFR_n, self.package, tr_loss=self.tr_loss,cv_loss=self.cv_loss), file_path)
                    print("Find better validated model, saving to %s" % file_path)

            # visualizing loss using visdom
            if self.visdom:
                x_axis = self.vis_epochs[0:epoch + 1]
                y_axis = torch.stack(
                    (self.tr_loss[0:epoch + 1], self.cv_loss[0:epoch + 1]), dim=1)
                if self.vis_window is None:
                    self.vis_window = self.vis.line(
                        X=x_axis,
                        Y=y_axis,
                        opts=self.vis_opts,
                    )
                else:
                    self.vis.line(
                        X=x_axis.unsqueeze(0).expand(y_axis.size(
                            1), x_axis.size(0)).transpose(0, 1),  # Visdom fix
                        Y=y_axis,
                        win=self.vis_window,
                        update='replace',
                    )

    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        # visualizing loss using visdom
        if self.visdom_epoch and not cross_valid:
            vis_opts_epoch = dict(title=self.visdom_id + " epoch " + str(epoch),
                                  ylabel='Loss', xlabel='Epoch')
            vis_window_epoch = None
            vis_iters = torch.arange(1, len(data_loader) + 1)
            vis_iters_loss = torch.Tensor(len(data_loader))

        my_timer = time.time()
        for i, (data) in enumerate(data_loader):
            padded_input, input_lengths, padded_target = data
            if self.args.is_npu:
                padded_input = padded_input.cpu()
                input_lengths = input_lengths.cpu()
                padded_target = padded_target.cpu()
            else:
                padded_input = padded_input.cuda()
                input_lengths = input_lengths.cuda()
                padded_target = padded_target.cuda()
            if not cross_valid:
                pred, gold = self.model(padded_input, input_lengths, padded_target)
                loss = cal_performance(pred, gold,
                                              smoothing=self.label_smoothing)
                self.optimizer.zero_grad()
                with amp.scale_loss(loss, self.optimizer.optimizer) as scaled_loss:
                    scaled_loss.backward()
                self.optimizer.step()
            else:
                pred, gold = self.model(padded_input, input_lengths, padded_target)
                loss = cal_performance(pred, gold,
                                              smoothing=self.label_smoothing)

            total_loss += loss.item()

            if i % self.print_freq == 0 and self.args.local_rank == 0:
                if cross_valid:
                    print('cross_valid | ', end='')
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} s/batch | local_rank {5} | FPS {6:.1f}'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), time.time() - my_timer, self.args.local_rank, self.args.batch_size * self.num_of_gpus / (time.time() - my_timer)),
                      flush=True)

            torch.npu.synchronize()
            my_timer = time.time()
            # visualizing loss using visdom
            if self.visdom_epoch and not cross_valid:
                vis_iters_loss[i] = loss.item()
                if i % self.print_freq == 0:
                    x_axis = vis_iters[:i+1]
                    y_axis = vis_iters_loss[:i+1]
                    if vis_window_epoch is None:
                        vis_window_epoch = self.vis.line(X=x_axis, Y=y_axis,
                                                         opts=vis_opts_epoch)
                    else:
                        self.vis.line(X=x_axis, Y=y_axis, win=vis_window_epoch,
                                      update='replace')

        return total_loss / (i + 1)
