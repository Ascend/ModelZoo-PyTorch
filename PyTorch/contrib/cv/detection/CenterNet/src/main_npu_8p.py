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

import _init_paths

import os

import torch
import torch.utils.data
if torch.__version__ >= "1.8":
    import torch_npu
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from lib.datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from apex import amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import apex


def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map


def main(opt, qtepoch=[0,]):
  if opt.precision_mode == 'must_keep_origin_dtype':
        option = {}
        option["ACL_PRECISION_MODE"] = "must_keep_origin_dtype" 
        torch.npu.set_option(option)
        torch.npu.config.allow_internal_format=False
  if opt.bin_model != 0:
        torch.npu.set_compile_mode(jit_compile=False) 
        print("use bin train model")
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  if opt.local_rank ==0:
    print(opt)
  os.environ['MASTER_ADDR'] = opt.addr
  os.environ['MASTER_PORT'] = opt.port
  device_id = int(opt.device_list.split(',')[int(opt.local_rank)])
  opt.device = 'npu:{}'.format(device_id)
 
  torch.npu.set_device(opt.device)
  print(opt.device)
  
  logger = Logger(opt)
  device_map = device_id_to_process_device_map(opt.device_list)  
  nproc_per_node = len(device_map) 
  global_rank = opt.rank * nproc_per_node +  opt.local_rank 
  dist.init_process_group(backend='hccl', world_size=opt.world_size, rank=global_rank)
  print('process{},device:{}'.format(opt.local_rank,opt.device))
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv, opt.load_local_weights, opt.local_weights_path)
  model = model.to(opt.device) #npu
  if opt.pretrained:
      checkpoint = torch.load(opt.pretrained_weight_path, map_location='cpu')
      if 'module.' in list(checkpoint['state_dict'].keys())[0]:
          checkpoint['state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
      model.load_state_dict(checkpoint['state_dict'], strict=False)

  # optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  if opt.precision_mode == 'must_keep_origin_dtype':
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O0", combine_grad=False) ###npu
  else:
    optimizer = apex.optimizers.NpuFusedAdam(model.parameters(), opt.lr) 
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1",loss_scale=19.0,combine_grad=True) ###npu
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr,   opt.lr_step)
  print('start_epoch:{}'.format(start_epoch))
  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.device_list, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=False
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return
  
  train_sampler = torch.utils.data.distributed.DistributedSampler(Dataset(opt, 'train'))
  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      num_workers=opt.num_workers,
      pin_memory=False,
      drop_last=True,
      shuffle=(train_sampler is None),
      sampler=train_sampler,
  )

  print('Starting training...')
  best = 1e10
  ###prof
  if opt.debug_prof:
    opt.num_epochs = 1
  ###prof
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    qtepoch.append(epoch)
    train_sampler.set_epoch(epoch)
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    if opt.local_rank == 0:
      logger.write('epoch: {} |'.format(epoch))
      for k, v in log_dict_train.items():
        logger.scalar_summary('train_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))

      if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
        save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                   epoch, model, optimizer)
        # with torch.no_grad():
        #   log_dict_val, preds = trainer.val(epoch, val_loader)
        # for k, v in log_dict_val.items():
        #   logger.scalar_summary('val_{}'.format(k), v, epoch)
        #   logger.write('{} {:8f} | '.format(k, v))
      
      print('best:{} metric:{}  epochs:{}'.format(best,log_dict_train[opt.metric],epoch),flush=True)
      
      if log_dict_train[opt.metric] < best:
        best = log_dict_train[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                    epoch, model)
      else:
        save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                  epoch, model, optimizer)
      logger.write('\n')

    if epoch in opt.lr_step:
      if opt.local_rank == 0:
        save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      if opt.local_rank == 0:
        print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
