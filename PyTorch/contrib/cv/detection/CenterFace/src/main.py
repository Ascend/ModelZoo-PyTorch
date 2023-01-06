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
import torch.utils.data
from opts_pose import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from datasets.sample.multi_pose import Multiposebatch
from apex import amp
if torch.__version__ >= "1.8":
    import torch_npu

def main(opt, qtepoch=[0,]):

  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  opt.device=f'npu:' + opt.device_list
  torch.npu.set_device(opt.device)
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  model = model.to(opt.device)  
  if opt.pretrained:
      checkpoint = torch.load(opt.pretrained_weight_path, map_location='cpu')
      if 'module.' in list(checkpoint['state_dict'].keys())[0]:
          checkpoint['state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
      model.load_state_dict(checkpoint['state_dict'], strict=False)

  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  model, optimizer = amp.initialize(model, optimizer, opt_level="O1",loss_scale=19.0)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
  print('Setting up data...')
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
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True,
      collate_fn=Multiposebatch
  )

  print('Starting training...')
  best = 1e10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    qtepoch.append(epoch)
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
    print(" ")
    print(" ")
    print(" ")
    print('best:{} metric:{}  epotchs:{}'.format(best,log_dict_train[opt.metric],epoch))
    print(" ")
    print(" ")
    print(" ")
    if log_dict_train[opt.metric] < best:
      best = log_dict_train[opt.metric]
      save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                   epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
