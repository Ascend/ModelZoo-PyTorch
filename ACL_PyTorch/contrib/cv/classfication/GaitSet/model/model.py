# -*- coding: utf-8 -*-
# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License


import math
import os
import os.path as osp
import random
import sys
from datetime import datetime
# from apex import amp
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata

import argparse
import shutil
import warnings
import time

import torch.distributed

from .network import TripletLoss, SetNet
from .utils import TripletSampler

class wrapperNet(nn.Module):
    def __init__(self, module):
        super(wrapperNet, self).__init__()
        self.module = module

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', start_count_index=10):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.start_count_index = start_count_index

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count == 0:
            self.N = n

        self.val = val
        self.count += n
        if self.count > (self.start_count_index * self.N):
            self.sum += val * n
            self.avg = self.sum / (self.count - self.start_count_index * self.N)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, n_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(n_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print(entries)

    def _get_batch_fmtstr(self, n_batches):
        n_digits = len(str(n_batches // 1))
        fmt = '{:' + str(n_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(n_batches) + ']'

class Model:
    def __init__(self,
                 hidden_dim,
                 lr,
                 hard_or_full_trip,
                 margin,
                 num_workers,
                 batch_size,
                 restore_iter,
                 total_iter,
                 save_name,
                 train_pid_num,
                 frame_num,
                 model_name,
                 train_source,
                 test_source,
                 img_size=64):

        self.save_name = save_name
        self.train_pid_num = train_pid_num
        self.train_source = train_source
        self.test_source = test_source

        self.hidden_dim = hidden_dim
        self.lr = lr
        self.hard_or_full_trip = hard_or_full_trip
        self.margin = margin
        self.frame_num = frame_num
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model_name = model_name
        self.P, self.M = batch_size

        self.restore_iter = restore_iter
        self.total_iter = total_iter

        self.img_size = img_size
        
        use_dist = False
        '''
        try:
            local_rank = torch.distributed.get_rank()
        except AssertionError:  # Default process group is not initialized
            use_dist = False
        '''
        
        if use_dist:
            self.encoder = SetNet(self.hidden_dim).float()
            self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
            self.optimizer = optim.Adam([
                {'params': self.encoder.parameters()},
            ], lr=self.lr)
            
            self.local_device = f'npu:{local_rank}'
            self.encoder.to(self.local_device)
            self.triplet_loss.to(self.local_device)
            
            # self.encoder,self.optimizer = amp.initialize(self.encoder,self.optimizer,opt_level="O2", loss_scale=32.0)
            
            local_rank = torch.distributed.get_rank()
            if torch.npu.device_count() > 1:
                print("Let's use",torch.npu.device_count(),"NPUs!")
                print('-----RANK=', local_rank)
                self.encoder = nn.parallel.DistributedDataParallel(self.encoder, broadcast_buffers=False, device_ids=[local_rank])
        else:
            self.local_device = 'npu'
            self.encoder = SetNet(self.hidden_dim).float()
            self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
            self.triplet_loss = nn.DataParallel(self.triplet_loss)
            self.encoder = self.encoder.npu()
            self.triplet_loss = self.triplet_loss.npu()
            
            self.optimizer = optim.Adam([
                {'params': self.encoder.parameters()},
            ], lr=self.lr)
            
            # self.encoder,self.optimizer = amp.initialize(self.encoder,self.optimizer,opt_level="O2", loss_scale=64.0)
            self.encoder = nn.DataParallel(self.encoder)
        
        self.hard_loss_metric = []
        self.full_loss_metric = []
        self.full_loss_num = []
        self.dist_list = []
        self.mean_dist = 0.01

        self.sample_type = 'all'

    def collate_fn(self, batch):
        batch_size = len(batch)
        feature_num = len(batch[0][0])
        seqs = [batch[i][0] for i in range(batch_size)]
        frame_sets = [batch[i][1] for i in range(batch_size)]
        view = [batch[i][2] for i in range(batch_size)]
        seq_type = [batch[i][3] for i in range(batch_size)]
        label = [batch[i][4] for i in range(batch_size)]
        
        batch = [seqs, view, seq_type, label, None]
        
        def select_frame(index):
            sample = seqs[index]
            frame_set = frame_sets[index]
            
            if self.sample_type == 'random':
                frame_list = sorted(list(frame_set))
                
                frame_id_list = random.choices(frame_list, k=self.frame_num)
                
                _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.values for feature in sample]
            return _
        
        seqs = list(map(select_frame, range(len(seqs))))

        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        else:
            npu_num = min(torch.npu.device_count(), batch_size)
            batch_per_npu = math.ceil(batch_size / npu_num)
            batch_frames = [[
                                len(frame_sets[i])
                                for i in range(batch_per_npu * _, batch_per_npu * (_ + 1))
                                if i < batch_size
                                ] for _ in range(npu_num)]
            if len(batch_frames[-1]) != batch_per_npu:
                for _ in range(batch_per_npu - len(batch_frames[-1])):
                    batch_frames[-1].append(0)
            max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(npu_num)])
            seqs = [[
                        np.concatenate([
                                           seqs[i][j]
                                           for i in range(batch_per_npu * _, batch_per_npu * (_ + 1))
                                           if i < batch_size
                                           ], 0) for _ in range(npu_num)]
                    for j in range(feature_num)]
            seqs = [np.asarray([
                                   np.pad(seqs[j][_],
                                          ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                                          'constant',
                                          constant_values=0)
                                   for _ in range(npu_num)])
                    for j in range(feature_num)]
            batch[4] = np.asarray(batch_frames)

        batch[0] = seqs
        return batch

    def fit(self):
        is_8p = torch.npu.device_count() > 1
        
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        hard_loss_mean = AverageMeter('Hard_Loss', ':.6e', start_count_index=0)
        full_loss_mean = AverageMeter('Full_Loss', ':.6e', start_count_index=0)
        p_full_loss_num = AverageMeter('Full_Loss_Num', ':6.3e', start_count_index=0)
        
        if self.restore_iter != 0:
            self.load(self.restore_iter)

        self.encoder.train()
        self.sample_type = 'random'
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        
        local_rank = 'npu'
        try:
            local_rank = torch.distributed.get_rank()
        except AssertionError:
            pass
        
        triplet_sampler = TripletSampler(self.train_source, self.batch_size)
        
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            # shuffle=False,
            # batch_size=self.batch_size,
            # pin_memory=False,
            batch_sampler=triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)
        train_label_set = list(self.train_source.label_set)
        train_label_set.sort()
        _time1 = datetime.now()
        
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, hard_loss_mean, full_loss_mean, p_full_loss_num],
            prefix="Iter[{}]".format(self.restore_iter))
        start_time = time.time()
        
        for iter_i, _t_data in enumerate(train_loader):
            data_time.update(time.time() - start_time)
            
            seq, view, seq_type, label, batch_frame = _t_data
            # triplet_sampler.set_epoch(self.restore_iter)
            
            self.restore_iter += 1
            self.optimizer.zero_grad()
            
            for i in range(len(seq)):
                seq[i] = self.np2var(seq[i]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()
            
            feature = self.encoder(*seq, batch_frame)
            
            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.array(target_label)).long()

            triplet_feature = feature.permute(1, 0, 2).contiguous()
            triplet_label = target_label.unsqueeze(0).cpu().repeat(triplet_feature.size(0), 1)
            
            triplet_feature = triplet_feature.npu()
            triplet_label = triplet_label.npu()
            
            (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num
             ) = self.triplet_loss(triplet_feature, triplet_label)
            if self.hard_or_full_trip == 'hard':
                loss = hard_loss_metric.mean()
            elif self.hard_or_full_trip == 'full':
                loss = full_loss_metric.mean()
            
            self.hard_loss_metric.append(hard_loss_metric.mean().data.cpu().numpy())
            self.full_loss_metric.append(full_loss_metric.mean().data.cpu().numpy())
            self.full_loss_num.append(full_loss_num.mean().data.cpu().numpy())
            self.dist_list.append(mean_dist.mean().data.cpu().numpy())
            
            if loss > 1e-9:
                # with amp.scale_loss(loss,self.optimizer) as scaled_loss:
                #     scaled_loss.backward()
                loss.backward()
                self.optimizer.step()

            if self.restore_iter % 1000 == 0:
                print(f"[{local_rank}]:", datetime.now() - _time1)
                _time1 = datetime.now()

            if self.restore_iter % 10 == 0:
                print(f"[{local_rank}]: ", 'iter {}:'.format(self.restore_iter), end='')
                
                self.mean_dist = np.mean(self.dist_list)
                print('mean_dist={0:.8f}'.format(self.mean_dist))
                
                hard_loss_mean.update(np.mean(self.hard_loss_metric), self.P * self.M)
                full_loss_mean.update(np.mean(self.full_loss_metric), self.P * self.M)
                p_full_loss_num.update(np.mean(self.full_loss_num), self.P * self.M)
                progress.display(self.restore_iter)
                
                sys.stdout.flush()
                self.hard_loss_metric = []
                self.full_loss_metric = []
                self.full_loss_num = []
                self.dist_list = []
            
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            
            if self.restore_iter % 200 == 0:
                self.save()

            if self.restore_iter == self.total_iter:
                break
    
    def ts2var(self, x):
        return autograd.Variable(x).to(self.local_device, non_blocking=False)
    
    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))
    
    def transform(self, flag, batch_size=1, pre_process=False, post_process=False):
        self.encoder.eval()
        source = self.test_source if flag == 'test' else self.train_source
        self.sample_type = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()
        
        test_len = len(data_loader)
        
        for i, x in enumerate(data_loader):
            import time
            cvt_time = time.time()
            
            seq, view, seq_type, label, batch_frame = x
            for j in range(len(seq)):
                seq[j] = self.np2var(seq[j]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()
            
            if pre_process:
                bin_dir_path = os.path.abspath('./')
                bin_img_path = os.path.join(bin_dir_path, f'../CASIA-B-bin/{i:0>4d}.bin')
                
                if i == 0:
                    print(f'seq[0].shape={seq[0].shape}')
                align_size = 100
                
                # new pre-process align by repeat itself
                cat_seq = None
                seq[0] = seq[0].detach().cpu().float()
                org_size = seq[0].shape[1]
                if org_size < align_size:
                    pad_shape = list(seq[0].shape)
                    pad_shape[1] = align_size - org_size
                    if i == 0:
                        print(f'pad_shape={pad_shape}')
                    pad_zeros = torch.zeros(pad_shape).float()
                    cat_seq = torch.cat([pad_zeros.float(), seq[0].float()], dim=1)
                    if i == 0:
                        print(f'cat_seq.shape={cat_seq.shape}')
                else:
                    cat_seq = seq[0].float()
                    while cat_seq.shape[1] < align_size:
                        cat_seq = torch.cat([cat_seq, seq[0].float()], dim=1)
                    cat_seq = cat_seq[:, :align_size, :, :]
                
                cat_seq.numpy().tofile(bin_img_path)
                
                if i % 50 ==0:
                    print(f'{i:0>4d}/{test_len}.bin Converted. Cost:{time.time()-cvt_time}(s)/img')
                
                continue  # pre-processing, skip model calculation
            
            # add post_process
            feature = None
            if post_process == False:
                feature = self.encoder(*seq, batch_frame)
            else:
                feat = np.fromfile(f'../result/dumpOutput_device0/{i:0>4d}_1.bin', dtype=np.float32)
                feature = torch.Tensor(feat).float().npu().view(1, -1, 256)

                if i % 50 == 0:
                    print(f'{i} feature read.')
            
            n, num_bin, _ = feature.size()
            feature_list.append(feature.view(n, -1).data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list += label
        if pre_process:
            return None
        return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list

    def save(self):
        os.makedirs(osp.join('checkpoint', self.model_name), exist_ok=True)
        local_rank = torch.distributed.get_rank()
        if local_rank != 0:
            return
        torch.save(self.encoder.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-encoder.ptm'.format(
                                self.save_name, self.restore_iter)))
        torch.save(self.optimizer.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-optimizer.ptm'.format(
                                self.save_name, self.restore_iter)))

    # restore_iter, iteration index of the checkpoint to load
    def load(self, restore_iter):
        self.optimizer.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-optimizer.ptm'.format(self.save_name, restore_iter)), map_location=torch.device('cpu')))
        try:
            self.encoder.load_state_dict(torch.load(osp.join(
                'checkpoint', self.model_name,
                '{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter)), map_location=torch.device('cpu')))
        except RuntimeError:
            wrapped = wrapperNet(self.encoder)
            wrapped.load_state_dict(torch.load(osp.join(
                'checkpoint', self.model_name,
                '{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter)), map_location=torch.device('cpu')))
            self.encoder = wrapped.module
