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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import torch
import torch.npu
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter
from torch.nn.parallel import DistributedDataParallel as DDP
from apex import amp
try:
    from torch_npu.utils.profiler import Profile
except ImportError:
    print("Profile not in torch_npu.utils.profiler now.. Auto Profile disabled.", flush=True)
    class Profile:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def end(self):
            pass


class NoProling(object):
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class ModelWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModelWithLoss, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, batch):
    outputs = self.model(batch['input'])
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats

class BaseTrainer(object):
  def __init__(
    self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses(opt)
    self.model_with_loss = ModelWithLoss(model, self.loss)

  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus)>1:
        self.model_with_loss = DDP(self.model_with_loss,device_ids=[device],broadcast_buffers=False)
    else:
      self.model_with_loss = self.model_with_loss.to(device)

    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

  def run_epoch(self, phase, epoch, data_loader):
    model_with_loss = self.model_with_loss
    if phase == 'train':
      model_with_loss.train()
    else:
      if len(self.opt.device_list) > 1:
        model_with_loss = self.model_with_loss.module
      model_with_loss.eval()
      torch.npu.empty_cache()  # npu change

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()
    fps = 0
    avg_time = 0
    profiler = Profile(start_step=int(os.getenv("PROFILE_START_STEP", 10)), profile_type=os.getenv("PROFILE_TYPE"))
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      # step fps
      per_step_time_start=time.time()
      ###FPS
      if iter_id == 5:
        start_time = time.time()
      # FPS
      data_time.update(time.time() - end)
      profiler.start()
      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)
      output, loss, loss_stats = model_with_loss(batch)
      loss = loss.mean()
      if phase == 'train':
        self.optimizer.zero_grad()
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        self.optimizer.step()
        profiler.end()
        batch_time.update(time.time() - end)
        end = time.time()

        Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
          epoch, iter_id, num_iters, phase=phase,
          total=bar.elapsed_td, eta=bar.eta_td)
        for l in avg_loss_stats:
          avg_loss_stats[l].update(
            loss_stats[l].mean().item(), batch['input'].size(0))
          Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
        if not opt.hide_data_time:
          Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
            '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
        if opt.print_iter > 0:
          if iter_id % opt.print_iter == 0:
            print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
        else:
          bar.next()

        if opt.debug > 0:
          self.debug(batch, output, iter_id)

        if opt.test:
          self.save_result(output, batch, results)
        del output, loss, loss_stats
        # step fps
        per_step_time = time.time() - per_step_time_start
        print('iter = {}, batch_size = {}, iter_time = {}, iter_fps = {}'.format(iter_id,
              opt.batch_size, per_step_time, opt.batch_size/per_step_time))
        ###FPS
        if iter_id == (len(data_loader)-1) and opt.local_rank ==0:
          all_time =(time.time()-start_time)
          avg_time = all_time/(len(data_loader)-5)
          if opt.world_size == -1:
            fps =  opt.batch_size/avg_time
          else:
            fps =  opt.batch_size * opt.world_size/avg_time
          print('')
          print('epoch = {}, all_time = {} ,avg_time = {}, batch_size = {}, FPS = {}'.format(
              epoch ,all_time,avg_time,opt.batch_size,fps))
          print('')
        ###FPS

    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    ###fps
    if opt.local_rank == 0:
      ret['avg_time'] = avg_time
      ret['FPS'] = fps
    ###fps

    return ret, results

  def debug(self, batch, output, iter_id):
    raise NotImplementedError

  def save_result(self, output, batch, results):
    raise NotImplementedError

  def _get_losses(self, opt):
    raise NotImplementedError

  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)
