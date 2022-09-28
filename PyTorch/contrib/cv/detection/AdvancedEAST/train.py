# Copyright 2021 Huawei Technologies Co., Ltd
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

import time

import torch
if torch.__version__>= '1.8':
    import torch_npu
import torch.utils.data
import torch.optim as optim
from torch import distributed as dist

from model_VGG import advancedEAST
from losses import quad_loss
from dataset import RawDataset, data_collate
from utils import Averager
import cfg

device = torch.device(cfg.device)


def train():
    if cfg.distributed:
        torch.npu.set_device(cfg.local_rank)
        dist.init_process_group(backend='hccl', world_size=cfg.world_size, rank=cfg.local_rank)

    """ dataset preparation """
    train_dataset = RawDataset(is_val=False)
    val_dataset = RawDataset(is_val=True)

    if cfg.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        collate_fn=data_collate,
        shuffle=(train_sampler is None),
        num_workers=int(cfg.workers),
        pin_memory=True,
        sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        collate_fn=data_collate,
        shuffle=False,
        num_workers=int(cfg.workers),
        pin_memory=True,
        sampler=val_sampler)

    # --------------------训练过程---------------------------------
    model = advancedEAST()
    if cfg.pth_path:
        model.load_state_dict(torch.load(cfg.pth_path, map_location='cpu'))
        if cfg.is_master_node:
            print('Load {}'.format(cfg.pth_path))
    elif int(cfg.train_task_id[-3:]) != 256:
        id_num = cfg.train_task_id[-3:]
        idx_dic = {'384': 256, '512': 384, '640': 512, '736': 640}
        state_dict = {k.replace('module.', ''): v for k, v in torch.load(
            './saved_model/3T{}_latest.pth'.format(idx_dic[id_num]), map_location='cpu').items()}
        model.load_state_dict(state_dict)
        if cfg.is_master_node:
            print('Load ./saved_model/3T{}_latest.pth'.format(idx_dic[id_num]))

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.decay)

    # Apex
    if cfg.amp:
        import apex
        optimizer = apex.optimizers.NpuFusedAdam(model.parameters(), lr=cfg.lr, weight_decay=cfg.decay)
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1", loss_scale="dynamic", combine_grad=True)

    if cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank], broadcast_buffers=False)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epoch_num)

    loss_func = quad_loss

    '''start training'''
    start_iter = 0
    start_time = time.time()
    i = start_iter
    step_num = 0
    loss_avg = Averager()
    val_loss_avg = Averager()
    total_train_img = int(cfg.total_img * (1 - cfg.validation_split_ratio))

    while(True):
        model.train()
        if cfg.distributed:
            train_sampler.set_epoch(i)
        # train part
        # training-----------------------------
        epoch_start_time = time.time()
        for image_tensors, labels, gt_xy_list in train_loader:
            step_num += 1
            batch_x = image_tensors.float().to(device)
            batch_y = labels.float().to(device)  # float64转float32
            out = model(batch_x)
            loss = loss_func(batch_y, out)
            optimizer.zero_grad()
            if cfg.amp:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            loss_avg.add(loss)

        loss = loss_avg.val()
        if cfg.distributed:
            dist.all_reduce(loss)
            loss = loss / cfg.world_size
        loss = loss.item()

        if cfg.is_master_node:
            print('Epoch:[{}/{}] Training loss:{:.3f} FPS:{:.3f} LR:{:.3e}'.format(i + 1, cfg.epoch_num,
                loss, total_train_img / (time.time() - epoch_start_time), optimizer.param_groups[0]['lr']))
        loss_avg.reset()

        scheduler.step()

        # evaluation--------------------------------
        if (i + 1) % cfg.val_interval == 0:
            elapsed_time = time.time() - start_time
            if cfg.is_master_node:
                print('Elapsed time:{}s'.format(round(elapsed_time)))
            model.eval()
            for image_tensors, labels, gt_xy_list in valid_loader:
                batch_x = image_tensors.float().to(device)
                batch_y = labels.float().to(device)  # float64转float32

                out = model(batch_x)
                loss = loss_func(batch_y, out)

                val_loss_avg.add(loss)

            loss = val_loss_avg.val()
            if cfg.distributed:
                dist.all_reduce(loss)
                loss = loss / cfg.world_size
            loss = loss.item()

            if cfg.is_master_node:
                print('Validation loss:{:.3f}'.format(loss))
            val_loss_avg.reset()

        if i + 1 == cfg.epoch_num:
            if cfg.is_master_node:
                torch.save(model.state_dict(), './saved_model/{}_latest.pth'.format(cfg.train_task_id))
                print('End the training')
            break
        i += 1


if __name__ == '__main__':
    train()
