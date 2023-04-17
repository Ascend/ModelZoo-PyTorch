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

import os
import datetime

from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
from data_loader import get_loader
import torch.nn as nn
import torch
import torch.nn.functional as F
if torch.__version__ >= '1.8':
    import torch_npu
import numpy as np
import time
from apex import amp
import math
from averageMeter import AverageMeter
import matplotlib.pyplot as plt
try:
    from torch_npu.utils.profiler import Profile
except:
    print("Profile not in torch_npu.utils.profiler now.. Auto Profile disabled.", flush=True)
    class Profile:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def end(self):
            pass
import sys

def flush_print(func):
    def new_print(*args, **kwargs):
        func(*args, **kwargs)
        sys.stdout.flush()
    return new_print
print = flush_print(print)

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, config):
        """Initialize configurations."""
        self.epoch = config.epoch
        self.celeba_image_dir = config.celeba_image_dir
        self.attr_path = config.attr_path
        self.selected_attrs = config.selected_attrs
        self.celeba_crop_size = config.celeba_crop_size
        self.image_size = config.image_size
        self.batch_size = config.batch_size
        self.mode = config.mode
        self.num_workers = config.num_workers

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters


        # DDP.
        self.distributed = config.distributed
        self.world_size = config.npus

        # Amp.
        self.amp = config.amp

        # if not self.distributed else config.gpus

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

    def build_model(self, rank):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        if rank <= 0:
            self.print_network(self.G, 'G')
            self.print_network(self.D, 'D')

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.pth'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.pth'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))



    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x, device = "npu:0"):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None, device = "npu:0"):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            c_trg = c_org.clone()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.

            c_trg_list.append(c_trg.to(device))
        return c_trg_list

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

    def train(self, rank = -1, world_size = -1):
        """Train StarGAN within a single dataset."""
        epoch_time = AverageMeter("Epoch Time",':6.3f')
        self.rank = rank
        loc = 'npu:{}'.format(self.rank)
        torch.npu.set_device(loc)
        self.build_model(rank)
        print(torch.__version__)

        # FP32模式下Conv2D算子编译失败，添加变量走FP16
        option = {'ACL_PRECISION_MODE':'allow_fp32_to_fp16'}
        torch_npu.npu.set_option(option)
        
        ########################################################################################################################
        # Set DDP.
        # Use nccl for GPU, hccl for NPU.
        if self.distributed:
            torch.npu.set_device(rank)
            torch.distributed.init_process_group(backend="hccl", world_size = world_size, rank = rank)

        # Set data loader.
        if self.distributed:
            self.train_sampler, self.data_loader = get_loader(self.celeba_image_dir, self.attr_path, self.selected_attrs,
                                        self.celeba_crop_size, self.image_size, self.batch_size,
                                        'CelebA', self.mode, self.world_size, self.distributed)
        else:
            self.data_loader = get_loader(self.celeba_image_dir, self.attr_path, self.selected_attrs,
                                        self.celeba_crop_size, self.image_size, self.batch_size,
                                        'CelebA', self.mode, 8)

        self.G.to(loc)
        self.D.to(loc)

        if self.amp:
            print("====== use amp =======")
            self.G, self.g_optimizer = amp.initialize(self.G, self.g_optimizer, opt_level = "O1", loss_scale = 32.0)
            self.D, self.d_optimizer = amp.initialize(self.D, self.d_optimizer, opt_level = "O1", loss_scale = 32.0)

        if self.distributed:
            self.G = nn.parallel.DistributedDataParallel(self.G, device_ids = [rank], broadcast_buffers=False)
            self.D = nn.parallel.DistributedDataParallel(self.D, device_ids = [rank], broadcast_buffers=False)
        ########################################################################################################################
         
        # Fetch fixed inputs for debugging.
        data_iter = iter(self.data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(loc)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs, loc)
        
        D_Loss = []
        G_Loss = []

        # Start training from scratch or resume training.
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        if self.distributed:
            print('Start training...Rank:', rank)
        else:
            print("Everything goes well! 1P Start training")
        start_time = time.time()

        for epoch in range(self.epoch):

            if self.distributed:
                self.train_sampler.set_epoch(epoch)

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            profile_g = Profile(start_step=int(os.getenv('PROFILE_START_STEP', 10)),
                              profile_type=os.getenv('PROFILE_TYPE'))
            profile_d = Profile(start_step=int(os.getenv('PROFILE_START_STEP', 10)),
                              profile_type=os.getenv('PROFILE_TYPE'))

            for iters, (x_real, label_org) in enumerate(self.data_loader):
                
                if iters == 5:
                    epoch_start_time = time.time()
                if iters > 1000:
                    pass
                # Generate target domain labels randomly.
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]

                c_org = label_org.clone()
                c_trg = label_trg.clone()

                x_real = x_real.to(loc)           # Input images.
                c_org = c_org.to(loc)             # Original domain labels.
                c_trg = c_trg.to(loc)             # Target domain labels.
                label_org = label_org.to(loc)     # Labels for computing classification loss.
                label_trg = label_trg.to(loc)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

                # Compute loss with real images.
                profile_d.start()
                out_src, out_cls = self.D(x_real)
                d_loss_real = - torch.mean(out_src)
                d_loss_cls = self.classification_loss(out_cls, label_org)

                # Compute loss with fake images.
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(loc)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat, loc)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
                self.reset_grad()
                if self.amp:
                    with amp.scale_loss(d_loss, self.d_optimizer) as scaled_loss:
                        scaled_loss.backward()  
                else:
                    d_loss.backward()
                self.d_optimizer.step()
                profile_d.end()

                # Logging.
                loss = {}
                loss['D Loss'] = d_loss.item()

                
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
                if (iters+1) % self.n_critic == 0:
                    profile_g.start()
                    # Original-to-target domain.
                    x_fake = self.G(x_real, c_trg)
                    out_src, out_cls = self.D(x_fake)
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_cls = self.classification_loss(out_cls, label_trg)

                    # Target-to-original domain.
                    x_reconst = self.G(x_fake, c_org)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    self.reset_grad()
                    if self.amp:
                        with amp.scale_loss(g_loss, self.g_optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        g_loss.backward()
                    self.g_optimizer.step()
                    profile_g.end()

                    # Logging.
                    loss["G Loss"] = g_loss.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #
                if rank <= 0:
                    # Print out training information.
                    if (iters+1) % self.log_step == 0:
                        et = time.time() - start_time
                        et = str(datetime.timedelta(seconds=et))[:-7]
                        log = "Elapsed [{}], Epoch [{}], Iteration [{}/{}],     ".format(et, epoch + 1, 
                        iters + 1, self.data_loader.dataset.__len__() // self.batch_size // self.world_size)
                        for tag, value in loss.items():
                            log += "{}: {:.4f}     ".format(tag, value)
                        print(log)
                        D_Loss.append(d_loss.item())
                        G_Loss.append(g_loss.item())

            epoch_time.update(time.time() - epoch_start_time)
            if epoch_time.avg > 0:
                content = "[NPU:" + str(self.rank) + "]" + 'FPS@all {:.3f}, TIME@all {:.3f}'.format(self.data_loader.dataset.__len__() / epoch_time.avg, epoch_time.avg)
                print(content)
                datapath = os.path.join(self.log_dir, 'FPS_Log.txt')
                with open(datapath, 'a') as f:
                    f.write(content)

            if rank <= 0:
                plt.figure(figsize=(20,10))
                if epoch_time.avg > 0:
                    plt.title('FPS@all {:.3f}, TIME@all {:.3f}'.format(self.batch_size * self.world_size / epoch_time.avg, epoch_time.avg))
                plt.subplot(121)
                plt.title("G Loss")
                plt.plot(G_Loss)
                plt.subplot(122)
                plt.title("D Loss")
                plt.plot(D_Loss)
                plt.draw()
                plt.savefig(os.path.join(self.log_dir, 'GAN_Loss.jpg'))
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.G(x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(epoch+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))
                if (epoch + 1) % 10 == 0:
                    G_path = os.path.join(self.model_save_dir, '{}-G.pth'.format(epoch + 1))
                    D_path = os.path.join(self.model_save_dir, '{}-D.pth'.format(epoch + 1))
                    torch.save(self.G.state_dict(), G_path)
                    torch.save(self.D.state_dict(), D_path)
                    print('Saved model checkpoints into {}...'.format(self.model_save_dir))

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.picture_number = 202599
        self.epoch_per_iter = math.ceil(self.picture_number / self.batch_size)
        self.restore_model(self.test_iters)
        
        # Set data loader.
        data_loader = self.celeba_loader

        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(loc)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs, loc)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))
