#
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
#
import torch
import apex
from torch.optim.adam import Adam
from model.ESRGAN import ESRGAN
from model.Discriminator import Discriminator
import os
from glob import glob
import torch.nn as nn
from torchvision.utils import save_image
from loss.loss import PerceptualLoss
import time
import torch.npu
import os
from apex import amp
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


class Trainer:
    def __init__(self, config, data_loader):
        self.device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')
        self.num_epoch = config.num_epoch
        self.epoch = config.epoch
        self.image_size = config.image_size
        self.data_loader = data_loader
        self.checkpoint_dir = config.checkpoint_dir
        self.batch_size = config.batch_size
        self.sample_dir = config.sample_dir
        self.nf = config.nf
        self.scale_factor = config.scale_factor
        self.apex = config.apex

        if config.is_perceptual_oriented:
            self.lr = config.p_lr
            self.content_loss_factor = config.p_content_loss_factor
            self.perceptual_loss_factor = config.p_perceptual_loss_factor
            self.adversarial_loss_factor = config.p_adversarial_loss_factor
            self.decay_iter = config.p_decay_iter
        else:
            self.lr = config.g_lr
            self.content_loss_factor = config.g_content_loss_factor
            self.perceptual_loss_factor = config.g_perceptual_loss_factor
            self.adversarial_loss_factor = config.g_adversarial_loss_factor
            self.decay_iter = config.g_decay_iter

        self.build_model()
        self.optimizer_generator = apex.optimizers.NpuFusedAdam(self.generator.parameters(), lr=self.lr, betas=(config.b1, config.b2),
                                        weight_decay=config.weight_decay)
        self.optimizer_discriminator = apex.optimizers.NpuFusedAdam(self.discriminator.parameters(), lr=self.lr, betas=(config.b1, config.b2),
                                            weight_decay=config.weight_decay)

        self.lr_scheduler_generator = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_generator, self.decay_iter)
        self.lr_scheduler_discriminator = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_discriminator, self.decay_iter)
        if config.apex:
            [self.generator, self.discriminator],[self.optimizer_generator,self.optimizer_discriminator] = \
            amp.initialize([self.generator, self.discriminator],[self.optimizer_generator,self.optimizer_discriminator],
                                              opt_level=config.apex_opt_level,
                                              loss_scale=config.loss_scale_value,
                                              combine_grad=True)

    def train(self):
        total_step = len(self.data_loader)
        adversarial_criterion = nn.BCEWithLogitsLoss().to(f'npu:{NPU_CALCULATE_DEVICE}')
        content_criterion = nn.L1Loss().to(f'npu:{NPU_CALCULATE_DEVICE}')
        perception_criterion = PerceptualLoss().to(f'npu:{NPU_CALCULATE_DEVICE}')
        self.generator.train()
        self.discriminator.train()
        
        n = 0
        for epoch in range(self.epoch, self.num_epoch):
            if not os.path.exists(os.path.join(self.sample_dir, str(epoch))):
                os.makedirs(os.path.join(self.sample_dir, str(epoch)))

            for step, image in enumerate(self.data_loader):
                if n == 200:
                    pass
                n += 1
                t = 0 
                start_time = time.time()
                low_resolution = image['lr'].to(f'npu:{NPU_CALCULATE_DEVICE}')
                high_resolution = image['hr'].to(f'npu:{NPU_CALCULATE_DEVICE}')

                real_labels = torch.ones((high_resolution.size(0), 1)).to(f'npu:{NPU_CALCULATE_DEVICE}')
                fake_labels = torch.zeros((high_resolution.size(0), 1)).to(f'npu:{NPU_CALCULATE_DEVICE}')

                ##########################
                #   training generator   #
                ##########################
                self.optimizer_generator.zero_grad()
                fake_high_resolution = self.generator(low_resolution)

                score_real = self.discriminator(high_resolution)
                score_fake = self.discriminator(fake_high_resolution)
                discriminator_rf = score_real - score_fake.mean()
                discriminator_fr = score_fake - score_real.mean()

                adversarial_loss_rf = adversarial_criterion(discriminator_rf, fake_labels)
                adversarial_loss_fr = adversarial_criterion(discriminator_fr, real_labels)
                adversarial_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2

                perceptual_loss = perception_criterion(high_resolution, fake_high_resolution)
                content_loss = content_criterion(fake_high_resolution, high_resolution)

                generator_loss = adversarial_loss * self.adversarial_loss_factor + \
                                 perceptual_loss * self.perceptual_loss_factor + \
                                 content_loss * self.content_loss_factor

                if self.apex:
                    with amp.scale_loss(generator_loss, self.optimizer_generator) as scaled_loss:
                        scaled_loss.backward()
                else:
                    generator_loss.backward()
                #generator_loss.backward()
                self.optimizer_generator.step()

                ##########################
                # training discriminator #
                ##########################

                self.optimizer_discriminator.zero_grad()

                score_real = self.discriminator(high_resolution)
                score_fake = self.discriminator(fake_high_resolution.detach())
                discriminator_rf = score_real - score_fake.mean()
                discriminator_fr = score_fake - score_real.mean()

                adversarial_loss_rf = adversarial_criterion(discriminator_rf, real_labels)
                adversarial_loss_fr = adversarial_criterion(discriminator_fr, fake_labels)
                discriminator_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2
                if self.apex:
                    with amp.scale_loss(discriminator_loss, self.optimizer_discriminator) as scaled_loss:
                        scaled_loss.backward()
                else:
                    discriminator_loss.backward()
                
                self.optimizer_discriminator.step()

                self.lr_scheduler_generator.step()
                self.lr_scheduler_discriminator.step()
                t += (time.time()-start_time)
                if step % 10 == 0:
                    print(f"[Epoch {epoch}/{self.num_epoch}] [Batch {step}/{total_step}] "
                          f"[D loss {discriminator_loss.item():.4f}] [G loss {generator_loss.item():.4f}] "
                          f"[adversarial loss {adversarial_loss.item() * self.adversarial_loss_factor:.4f}]"
                          f"[perceptual loss {perceptual_loss.item() * self.perceptual_loss_factor:.4f}]"
                          f"[content loss {content_loss.item() * self.content_loss_factor:.4f}]"
                          f"[time/step {t:.4f}]"
                          f"")
                    if step % 5000 == 0:
                        result = torch.cat((high_resolution, fake_high_resolution), 2)
                        save_image(result, os.path.join(self.sample_dir, str(epoch), f"SR_{step}.png"))

            torch.save(self.generator.state_dict(), os.path.join(self.checkpoint_dir, f"generator_{epoch}.pth"))
            torch.save(self.discriminator.state_dict(), os.path.join(self.checkpoint_dir, f"discriminator_{epoch}.pth"))

    def build_model(self):
        self.generator = ESRGAN(3, 3, 64, scale_factor=self.scale_factor).to(f'npu:{NPU_CALCULATE_DEVICE}')
        self.discriminator = Discriminator().to(f'npu:{NPU_CALCULATE_DEVICE}')
        self.load_model()

    def load_model(self):
        print(f"[*] Load model from {self.checkpoint_dir}")
        if not os.path.exists(self.checkpoint_dir):
            self.makedirs = os.makedirs(self.checkpoint_dir)

        if not os.listdir(self.checkpoint_dir):
            print(f"[!] No checkpoint in {self.checkpoint_dir}")
            return

        generator = glob(os.path.join(self.checkpoint_dir, f'generator_{self.epoch - 1}.pth'))
        discriminator = glob(os.path.join(self.checkpoint_dir, f'discriminator_{self.epoch - 1}.pth'))

        if not generator:
            print(f"[!] No checkpoint in epoch {self.epoch - 1}")
            return

        self.generator.load_state_dict(torch.load(generator[0]))
        self.discriminator.load_state_dict(torch.load(discriminator[0]))
