"""
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the BSD 3-Clause License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://spdx.org/licenses/BSD-3-Clause.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
from torch.optim import lr_scheduler

from opt import opt
from data import Data
from network import MGN
from loss import Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature
from utils.metrics import mean_ap, cmc, re_ranking
from utils.average_meter import AverageMeter
import time


# add
from apex import amp

if opt.device_num == -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
else:
    environ_str = '0'
    for i in range(1, opt.device_num):
        environ_str = environ_str + ',%d' % i
    os.environ["CUDA_VISIBLE_DEVICES"] = environ_str

if opt.npu:
    os.environ['MASTER_ADDR'] = opt.addr
    os.environ['MASTER_PORT'] = '29688'
    if opt.device_num > 1:
        torch.distributed.init_process_group(backend="hccl", rank=opt.local_rank, world_size=opt.device_num)
    torch.npu.manual_seed_all(opt.seed)
    torch.npu.set_device(opt.local_rank)
else:
    if opt.device_num > 1:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.manual_seed_all(opt.seed)
    torch.cuda.set_device(opt.local_rank)


class Main():
    def __init__(self, model, loss, data):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        self.testset = data.testset
        self.queryset = data.queryset

        if opt.npu:
            self.model = model.npu()
        else:
            self.model = model.cuda()
        self.loss = loss
        self.optimizer = get_optimizer(model)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1)
        
        # add
        if opt.npu:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O2", loss_scale=128.0, combine_grad=True)
        else:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O2", loss_scale=128.0)
        
        if opt.device_num > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, 
                        device_ids=[opt.local_rank], 
                        output_device=opt.local_rank,
                        find_unused_parameters=True,
                        broadcast_buffers=False
                        )

    def train(self, epoch):
        batch_time = AverageMeter('Time', ':6.3f')

        self.scheduler.step()

        self.model.train()
        end = time.time()
        for batch, (inputs, labels) in enumerate(self.train_loader):
            if opt.npu:
                inputs = inputs.npu()
                labels = labels.npu()
            else:
                inputs = inputs.cuda()
                labels = labels.cuda()
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            self.optimizer.step()
            batch_time.update(time.time() - end)
            if batch % 10 == 0:
                info = 'Epoch: [{N_epoch}][{N_batch}/{N_size}] | Time {N_bt:.3f} {N_bta:.3f} | Loss {N_loss:.3f}'.format(
                        N_epoch=epoch, N_batch=batch, N_size=len(self.train_loader),
                                N_bt=batch_time.val, N_bta=batch_time.avg,
                                N_loss=loss
                                )
                print(info)
            end = time.time()
        if opt.device_num <= 1 or opt.local_rank == 0:
            device_num = 1 if opt.device_num == -1 else opt.device_num
            print('\nFPS {:.3f}, TIME {:.3f}'.format(device_num * opt.batchid * opt.batchimage / batch_time.avg, batch_time.avg))
        return batch_time.avg

    def evaluate(self):

        self.model.eval()

        print('extract features, this may take a few minutes')
        qf = extract_feature(self.model, tqdm(self.query_loader)).numpy()
        gf = extract_feature(self.model, tqdm(self.test_loader)).numpy()

        def rank(dist):
            r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

            return r, m_ap

        #########################   re rank##########################
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

        r, m_ap = rank(dist)

        info = '[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'.format(m_ap, r[0], r[2], r[4], r[9])

        print(info)

        #########################no re rank##########################
        dist = cdist(qf, gf)

        r, m_ap = rank(dist)

        print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))
        
        return m_ap, info

    def vis(self):

        self.model.eval()

        gallery_path = data.testset.imgs
        gallery_label = data.testset.ids

        # Extract feature
        print('extract features, this may take a few minutes')
        query_feature = extract_feature(model, tqdm([(torch.unsqueeze(data.query_image, 0), 1)]))
        gallery_feature = extract_feature(model, tqdm(data.test_loader))

        # sort images
        query_feature = query_feature.view(-1, 1)
        score = torch.mm(gallery_feature, query_feature)
        score = score.squeeze(1).cpu()
        score = score.numpy()

        index = np.optort(score)  # from small to large
        index = index[::-1]  # from large to small

        # # Remove junk images
        # junk_index = np.argwhere(gallery_label == -1)
        # mask = np.in1d(index, junk_index, invert=True)
        # index = index[mask]

        # Visualize the rank result
        fig = plt.figure(figsize=(16, 4))

        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        plt.imshow(plt.imread(opt.query_image))
        ax.set_title('query')

        print('Top 10 images are as follow:')

        for i in range(10):
            img_path = gallery_path[index[i]]
            print(img_path)

            ax = plt.subplot(1, 11, i + 2)
            ax.axis('off')
            plt.imshow(plt.imread(img_path))
            ax.set_title(img_path.split('/')[-1][:9])

        fig.savefig("show.png")
        print('result saved to show.png')


if __name__ == '__main__':
    data = Data()
    model = MGN()
    loss = Loss()
    main = None
    if opt.mode != 'prof':
        main = Main(model, loss, data)

    if opt.mode == 'train':
        if opt.finetune:
            state_dict = torch.load(opt.weight)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:] # discard module.
                if k.startswith('classifier'):
                    continue
                new_state_dict[k] = v
            model.load_state_dict(new_state_dict)

        total_avg = 0.0
        best_dict = {
            'm_ap' : 0.0,
            'epoch' : -1,
            'info' : "",
        }

        for epoch in range(1, opt.epoch + 1):
            if opt.device_num <= 1 or opt.local_rank == 0:
                print('\nepoch', epoch)
            total_avg += main.train(epoch)
            if epoch % 50 == 0:
                print('\nstart evaluate')
                m_ap, info = main.evaluate()
                if best_dict['m_ap'] < m_ap:
                    best_dict['m_ap'] = m_ap
                    best_dict['epoch'] = epoch
                    best_dict['info'] = info

                if opt.device_num <= 1 or opt.local_rank == 0:
                    os.makedirs('weights', exist_ok=True)
                    torch.save(model.state_dict(), ('weights/model_{}.pt'.format(epoch)))
        print("--------------------------------------------------------")
        print("best mAP at epoch %d" % best_dict['epoch'])
        print("best evaluate info\n%s" % best_dict['info'])
        avg_time = total_avg / opt.epoch
        if opt.device_num <= 1 or opt.local_rank == 0:
            device_num = 1 if opt.device_num == -1 else opt.device_num
            print('FPS@all {:.3f}, TIME@all {:.3f}'.format(device_num * opt.batchimage * opt.batchid / avg_time, avg_time))

    if opt.mode == 'prof':
        if opt.npu:
            model = model.npu()
            optimizer = get_optimizer(model)
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=128.0, combine_grad=True)

            def run(data, loss):
                for index, (inputs, targets) in enumerate(data.train_loader):
                    out = model(inputs.npu())
                    loss1=loss(out, targets.int().npu())
                    optimizer.zero_grad()
                    with amp.scale_loss(loss1, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    optimizer.step()
                    break
            print("runing")
            run(data, loss)
            print("end")
        else:
            model = model.cuda()
            optimizer = get_optimizer(model)
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=128.0)

            def run(data, loss):
                for index, (inputs, targets) in enumerate(data.train_loader):
                    out = model(inputs.cuda())
                    loss1=loss(out, targets.cuda())
                    optimizer.zero_grad()
                    with amp.scale_loss(loss1, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    optimizer.step()
                    break
            for i in range(5):
                start_time = time.time()
                run(data, loss)
                print('iter: %d, time: %.2f'%(i, (time.time() - start_time)*1000))
            with torch.autograd.profiler.profile() as prof:
                run(data, loss)
            print(prof.key_averages().table())
            prof.export_chrome_trace("pytorch_prof_gpu.prof")

    if opt.mode == 'evaluate':
        print('start evaluate')
        model.load_state_dict(torch.load(opt.weight))
        main.evaluate()

    if opt.mode == 'vis':
        print('visualize')
        model.load_state_dict(torch.load(opt.weight))
        main.vis()
