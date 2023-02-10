# Copyright 2021 Huawei Technologies Co., Ltd
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter
from apex import amp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.npu
class ModleWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModleWithLoss, self).__init__()
    self.model = model
    self.loss = loss
  
  def forward(self, batch):   
    outputs = self.model(batch['input']) 
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats

class NoProfiling():
    def __enter__(self):
        ...
    def __exit__(self, exc_type, exc_val, exc_tb):
        ...

class BaseTrainer(object):
  def __init__(self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses(opt)
    self.model_with_loss = ModleWithLoss(model, self.loss)

  def set_device(self, gpus, chunk_sizes, device):
    #if len(gpus) > 1:
     # self.model_with_loss = DataParallel(
      #  self.model_with_loss, device_ids=gpus, 
       # chunk_sizes=chunk_sizes).to(device)
    if len(gpus)>1:
        self.model_with_loss = DDP(self.model_with_loss,device_ids=[device])  
    else:
      self.model_with_loss = self.model_with_loss.to(device)
      print('#########model_with_loss to device:{}#################'.format(device))
    
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
      torch.npu.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      if iter_id == 5:
        start_time = time.time()
      data_time.update(time.time() - end)
      
      if opt.start_step<= iter_id <= opt.stop_step and opt.profiling == 'True' and epoch==2:
        profiling = torch.npu.profile(profiler_result_path="./CANN_prof")
      else:
        profiling = NoProfiling()
      with profiling:
        for k in batch:
          if k != 'meta':
            batch[k] = batch[k].to(device=opt.device, non_blocking=True)
        
        output, loss, loss_stats = model_with_loss(batch)
        loss = loss.mean()
        if phase == 'train':
          self.optimizer.zero_grad()
          if opt.use_fp32:
            loss.backward()
          else:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
              scaled_loss.backward()
          self.optimizer.step()
        batch_time.update(time.time() - end)
        if iter_id < 3 and epoch==1:
          print('iter_time={}'.format(time.time() - end))
        end = time.time()
        if opt.local_rank ==0:
          Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
          epoch, iter_id, num_iters, phase=phase,
          total=bar.elapsed_td, eta=bar.eta_td)
        for l in avg_loss_stats:
          avg_loss_stats[l].update(
            loss_stats[l].mean().item(), batch['input'].size(0))
          if opt.local_rank ==0:
            Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
        if not opt.hide_data_time and opt.local_rank ==0:
          Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
            '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
        if opt.print_iter > 0 and opt.local_rank==0:
          if iter_id % opt.print_iter == 0:
            print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
        else:
          bar.next()
        
        if opt.debug > 0:
          self.debug(batch, output, iter_id)
        
        if opt.test:
          self.save_result(output, batch, results)
        del output, loss, loss_stats
        if iter_id == (len(data_loader)-1) and opt.local_rank ==0:        
          all_time =(time.time()-start_time)
          avg_time = all_time/(len(data_loader)-5)
          if opt.world_size == -1:
            fps = opt.batch_size/avg_time
          else:
            fps = opt.batch_size*opt.world_size/avg_time
          print('')
          print('all_time = {} ,avg_time = {}, batch_size = {}, FPS = {}'.format(all_time,avg_time,opt.batch_size,fps))
          print('')
          
    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
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
