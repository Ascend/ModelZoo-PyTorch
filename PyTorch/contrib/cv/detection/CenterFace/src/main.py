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

import _init_paths

import os

import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.utils.data
from opts_pose import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from datasets.sample.multi_pose import Multiposebatch
from apex import amp
import torch.npu
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def main(opt, qtepoch=[0,]):
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = opt.port
  if opt.bin_mode:
      torch.npu.set_compile_mode(jit_compile=False)
  if opt.use_fp32:
      option = {}
      option["ACL_PRECISION_MODE"] = "must_keep_origin_dtype"
      torch.npu.set_option(option)
      torch.npu.config.allow_internal_format=False

  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  if opt.local_rank ==0:
    print(opt)
  device_id = int(opt.device_list.split(',')[int(opt.local_rank)])
  opt.device = 'npu:{}'.format(device_id)

  torch.npu.set_device(opt.device)
  if opt.distributed_launch:
    dist.init_process_group(backend='hccl', world_size=opt.world_size, rank=opt.local_rank)

  print('process{},device:{}'.format(opt.local_rank,opt.device))
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  model = model.to(opt.device)
  if opt.pretrained:
      checkpoint = torch.load(opt.pretrained_weight_path, map_location='cpu')
      if 'module.' in list(checkpoint['state_dict'].keys())[0]:
          checkpoint['state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
      model.load_state_dict(checkpoint['state_dict'], strict=False)

  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  if not opt.use_fp32:
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1",loss_scale=19.0)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
  print('start_epoch:{}'.format(start_epoch))
  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  if opt.distributed_launch:
    trainer.set_device(opt.device_list, opt.chunk_sizes, opt.device)
  else:
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
  print('Setting up data...')
  if opt.distributed_launch:
    train_sampler = torch.utils.data.distributed.DistributedSampler(Dataset(opt, 'train'))
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'),
      batch_size=opt.batch_size,
      shuffle=False if opt.distributed_launch else True,
      num_workers=opt.num_workers,
      sampler=train_sampler if opt.distributed_launch else None,
      pin_memory=True,
      drop_last=True,
      collate_fn=Multiposebatch
  )

  print('Starting training...')
  best = 1e10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    qtepoch.append(epoch)
    if opt.distributed_launch:
        train_sampler.set_epoch(epoch)
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    if opt.local_rank == 0:
        str1 ='epoch:{}|'.format(epoch)
        for k, v in log_dict_train.items():
          str2 ='{} {:8f}|'.format(k,v)
          str1 = str1 +str2
        print(str1)
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
          save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                     epoch, model, optimizer)

        print('best:{} metric:{}  epotchs:{}'.format(best,log_dict_train[opt.metric],epoch))

        if log_dict_train[opt.metric] < best:
          best = log_dict_train[opt.metric]
          save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                       epoch, model)
        else:
          save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                     epoch, model, optimizer)

    if epoch in opt.lr_step:

        if opt.local_rank == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), epoch, model, optimizer)
        lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
        if opt.local_rank == 0:
            print('Drop LR to', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
