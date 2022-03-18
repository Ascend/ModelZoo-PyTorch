# Copyright 2020 Huawei Technologies Co., Ltd
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
import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import dataloader
import apex.amp as amp
import apex



class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32, class_num=10):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.class_num, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32, class_num=10):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim + self.class_num, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.conv(x)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x

class CGAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 62
        self.class_num = 10
        self.args = args
        self.sample_num = self.class_num ** 2

        # load dataset
        
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size, args=self.args)
        data = self.data_loader.__iter__().__next__()[0]

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size, class_num=self.class_num)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size, class_num=self.class_num)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.npu()
            self.D.npu()
            self.BCE_loss = nn.BCEWithLogitsLoss().npu()
        else:
            self.BCE_loss = nn.BCEWithLogitsLoss()
        if args.amp:
            self.G, self.G_optimizer = amp.initialize(self.G, self.G_optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale)
            self.D, self.D_optimizer = amp.initialize(self.D, self.D_optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale)
        if args.is_distributed:
            self.G = nn.parallel.DistributedDataParallel(self.G, device_ids=[args.local_rank], broadcast_buffers=False)
            self.D = nn.parallel.DistributedDataParallel(self.D, device_ids=[args.local_rank], broadcast_buffers=False)
        if args.local_rank == 0:
            print('---------- Networks architecture -------------')
        if args.local_rank == 0:
            utils.print_network(self.G)
        if args.local_rank == 0:
            utils.print_network(self.D)
        if args.local_rank == 0:
            print('-----------------------------------------------')

        # fixed noise & condition
        self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        for i in range(self.class_num):
            self.sample_z_[i*self.class_num] = torch.rand(1, self.z_dim)
            for j in range(1, self.class_num):
                self.sample_z_[i*self.class_num + j] = self.sample_z_[i*self.class_num]

        temp = torch.zeros((self.class_num, 1))
        for i in range(self.class_num):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(self.class_num):
            temp_y[i*self.class_num: (i+1)*self.class_num] = temp

        self.sample_y_ = torch.zeros((self.sample_num, self.class_num)).scatter_(1, temp_y.type(torch.LongTensor), 1)
        if self.gpu_mode:
            self.sample_z_, self.sample_y_ = self.sample_z_.npu(), self.sample_y_.npu()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.npu(), self.y_fake_.npu()

        self.D.train()
        if self.args.local_rank == 0:
            print('training start!!')
        start_time = time.time()
        sum_fps = 0
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, y_) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // (self.batch_size*self.args.num_gpus):
                    break
                if iter == 5:
                    time_iter5 = time.time()
                z_ = torch.rand((self.batch_size, self.z_dim))
                y_vec_ = torch.zeros((self.batch_size, self.class_num)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1)
                y_fill_ = y_vec_.unsqueeze(2).unsqueeze(3).expand(self.batch_size, self.class_num, self.input_size, self.input_size)
                if self.gpu_mode:
                    x_, z_, y_vec_, y_fill_ = x_.npu(), z_.npu(), y_vec_.npu(), y_fill_.npu()

                # update D network
                if self.args.is_distributed == 0 and epoch == 0 and iter == 6:
                    with torch.autograd.profiler.profile(use_npu=True) as prof: 
                        self.D_optimizer.zero_grad()

                        D_real = self.D(x_, y_fill_)
                        D_real_loss = self.BCE_loss(D_real, self.y_real_)

                        G_ = self.G(z_, y_vec_)
                        D_fake = self.D(G_, y_fill_)
                        D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                        D_loss = D_real_loss + D_fake_loss
                        self.train_hist['D_loss'].append(D_loss.item())

                        #D_loss.backward()
                        if self.args.amp:
                            with amp.scale_loss(D_loss, self.D_optimizer) as scaled_loss_D:
                                scaled_loss_D.backward()
                        else:
                            D_loss.backward()
                        self.D_optimizer.step()
                    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                    prof.export_chrome_trace("D_output.prof") # "output.prof"为输出文件地址
                else:
                    self.D_optimizer.zero_grad()

                    D_real = self.D(x_, y_fill_)
                    D_real_loss = self.BCE_loss(D_real, self.y_real_)

                    G_ = self.G(z_, y_vec_)
                    D_fake = self.D(G_, y_fill_)
                    D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                    D_loss = D_real_loss + D_fake_loss
                    self.train_hist['D_loss'].append(D_loss.item())

                    #D_loss.backward()
                    if self.args.amp:
                        with amp.scale_loss(D_loss, self.D_optimizer) as scaled_loss_D:
                            scaled_loss_D.backward()
                    else:
                        D_loss.backward()
                    self.D_optimizer.step()    
                # update G network
                if self.args.is_distributed == 0 and epoch == 0 and iter == 6:
                    with torch.autograd.profiler.profile(use_npu=True) as prof:
                        self.G_optimizer.zero_grad()

                        G_ = self.G(z_, y_vec_)
                        D_fake = self.D(G_, y_fill_)
                        G_loss = self.BCE_loss(D_fake, self.y_real_)
                        self.train_hist['G_loss'].append(G_loss.item())
                                        #G_loss.backward()
                        if self.args.amp:
                            with amp.scale_loss(G_loss, self.G_optimizer) as scaled_loss_G:
                                scaled_loss_G.backward()
                        else:
                            G_loss.backward()
                        self.G_optimizer.step()
                    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                    prof.export_chrome_trace("G_output.prof") # "output.prof"为输出文件地址
                else:
                    self.G_optimizer.zero_grad()
                    G_ = self.G(z_, y_vec_)
                    D_fake = self.D(G_, y_fill_)
                    G_loss = self.BCE_loss(D_fake, self.y_real_)
                    self.train_hist['G_loss'].append(G_loss.item())
                                    #G_loss.backward()
                    if self.args.amp:
                        with amp.scale_loss(G_loss, self.G_optimizer) as scaled_loss_G:
                            scaled_loss_G.backward()
                    else:
                        G_loss.backward()
                    self.G_optimizer.step()


                
                if (iter + 1) == self.data_loader.dataset.__len__() // (self.batch_size*self.args.num_gpus):
                    time_avg = time.time() - time_iter5
                    fps = self.args.num_gpus * self.args.batch_size * (self.data_loader.dataset.__len__() // (self.batch_size*self.args.num_gpus)) / time_avg
                    sum_fps += fps
                    avg_fps = sum_fps/(float(epoch + 1))
                    if self.args.local_rank == 0:
                        print("EPOCH: [%2d],FPS: %.2f,Average FPS: %2.f time: %.2f"%((epoch + 1), fps, avg_fps, time_avg))
                if ((iter + 1) % 100) == 0:
                    if self.args.local_rank == 0:
                        print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f G_loss: %.8f" %
                              ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // (self.batch_size*self.args.num_gpus), D_loss.item(), G_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                #self.visualize_results((epoch+1))
                if self.args.local_rank==0:
                    self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        if self.args.local_rank == 0:
            print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                  self.epoch, self.train_hist['total_time'][0]))
        if self.args.local_rank == 0:
            print("Training finish!... save training results")
        if self.args.local_rank==0:
            self.save()
        if self.args.local_rank == 0:
            utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                     self.epoch)
            utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        image_frame_dim = int(np.floor(np.sqrt(self.sample_num)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_, self.sample_y_)
        else:
            """ random noise """
            sample_y_ = torch.zeros(self.batch_size, self.class_num).scatter_(1, torch.randint(0, self.class_num - 1, (self.batch_size, 1)).type(torch.LongTensor), 1)
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_, sample_y_ = sample_z_.npu(), sample_y_.npu()

            samples = self.G(sample_z_, sample_y_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pth'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pth'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pth')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pth')))