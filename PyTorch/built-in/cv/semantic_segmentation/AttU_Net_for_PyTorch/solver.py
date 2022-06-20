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
import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import csv

try:
    import apex
    from apex import amp
except ImportError:
    amp = None

class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = torch.nn.BCELoss()
        self.augmentation_prob = config.augmentation_prob

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size
        self.config = config

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode

        self.device = torch.device('npu:'+str(config.npu_idx) if torch.npu.is_available() else 'cpu')
        torch.npu.set_device(self.device)
        self.model_type = config.model_type
        self.t = config.t
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type =='U_Net':
            self.unet = U_Net(img_ch=3,output_ch=1)
        elif self.model_type =='R2U_Net':
            self.unet = R2U_Net(img_ch=3,output_ch=1,t=self.t)
        elif self.model_type =='AttU_Net':
            if self.config.pretrained:
                print("=> using pre-trained model '{}'".format(self.model_type))
                self.unet = AttU_Net(img_ch=3,output_ch=1)
                print("Load my train models...")
                pretrained_dict = \
                torch.load(self.config.pth_path, map_location="cpu")
                self.unet.load_state_dict(pretrained_dict, strict=False) 
                print('%s is Successfully Loaded from %s'%(self.model_type,self.config.pth_path))        
            else:
                print("=> creating model '{}'".format(self.model_type))
                self.unet = AttU_Net(img_ch=3,output_ch=1)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=3,output_ch=1,t=self.t)
        self.unet.to(self.device)
        self.optimizer = apex.optimizers.NpuFusedAdam(list(self.unet.parameters()), self.lr, [self.beta1, self.beta2])

        if self.config.apex:
            self.unet, self.optimizer = amp.initialize(self.unet, self.optimizer,opt_level=self.config.apex_opt_level, loss_scale=self.config.loss_scale_value,combine_grad=True)

        if self.config.distributed:
            self.unet = torch.nn.parallel.DistributedDataParallel(self.unet, device_ids=[self.config.npu_idx])

        # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()

        if self.config.is_master_node:
            print(model)
            print(name)
            print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.npu.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def compute_accuracy(self,SR,GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

    def tensor2img(self,x):
        img = (x[:,0,:,:]>x[:,1,:,:]).float()
        img = img*255
        return img


    def train(self):
        """Train encoder, generator and discriminator."""

        #====================================== Training ===========================================#
        #===========================================================================================#
        
        unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))

        # Train for Encoder
        lr = self.lr
        best_unet_score = 0.

        for epoch in range(self.num_epochs):
            if self.config.distributed:
                self.config.train_sampler.set_epoch(epoch)

            self.unet.train(True)
            epoch_loss = 0

            acc = 0.    # Accuracy
            SE = 0.     # Sensitivity (Recall)
            SP = 0.     # Specificity
            PC = 0.     # Precision
            F1 = 0.     # F1 Score
            JS = 0.     # Jaccard Similarity
            DC = 0.     # Dice Coefficient
            length = 0
            start_time = 0
            steps = len(self.train_loader)
            for i, (images, GT) in enumerate(self.train_loader):
                # GT : Ground Truth
                if i == 5:
                    start_time = time.time()
                images = images.to(self.device,non_blocking=True)
                GT = GT.to(self.device,non_blocking=True)

                # SR : Segmentation Result
                SR = self.unet(images)
                SR_probs = F.sigmoid(SR)
                SR_flat = SR_probs.view(SR_probs.size(0),-1)

                GT_flat = GT.view(GT.size(0),-1)
                loss = self.criterion(SR_flat,GT_flat)
                epoch_loss += loss.item()

                # Backprop + optimize
                self.reset_grad()
                if self.config.apex:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                self.optimizer.step()

                acc += get_accuracy(SR,GT)
                SE += get_sensitivity(SR,GT)
                SP += get_specificity(SR,GT)
                PC += get_precision(SR,GT)
                F1 += get_F1(SR,GT)
                JS += get_JS(SR,GT)
                DC += get_DC(SR,GT)
                length += 1
                if i > 5 and self.config.is_master_node:
                    print('Epoch [%d/%d], Step:[%d/%d], Loss: %.4f, [Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
                            epoch + 1, self.num_epochs, i, steps, loss, \
                            acc, SE, SP, PC, F1, JS, DC,))
            FPS = self.batch_size * (steps - 5) * self.config.rank_size / (time.time() - start_time)
            acc = acc/length
            SE = SE/length
            SP = SP/length
            PC = PC/length
            F1 = F1/length
            JS = JS/length
            DC = DC/length

            # Print the log info
            if self.config.is_master_node:
                print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, FPS: %.2f' % (
                        epoch+1, self.num_epochs, \
                        epoch_loss,\
                        acc,SE,SP,PC,F1,JS,DC,FPS))

            # Decay learning rate
            if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
                lr -= (self.lr / float(self.num_epochs_decay))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                if self.config.is_master_node:
                    print ('Decay learning rate to lr: {}.'.format(lr))
 

            #===================================== Validation ====================================#
            self.unet.train(False)
            self.unet.eval()

            acc = 0.    # Accuracy
            SE = 0.     # Sensitivity (Recall)
            SP = 0.     # Specificity
            PC = 0.     # Precision
            F1 = 0.     # F1 Score
            JS = 0.     # Jaccard Similarity
            DC = 0.     # Dice Coefficient
            length=0
            for i, (images, GT) in enumerate(self.valid_loader):

                images = images.to(self.device)
                GT = GT.to(self.device)
                SR = F.sigmoid(self.unet(images))
                acc += get_accuracy(SR,GT)
                SE += get_sensitivity(SR,GT)
                SP += get_specificity(SR,GT)
                PC += get_precision(SR,GT)
                F1 += get_F1(SR,GT)
                JS += get_JS(SR,GT)
                DC += get_DC(SR,GT)
                        
                length += 1
                    
            acc = acc/length
            SE = SE/length
            SP = SP/length
            PC = PC/length
            F1 = F1/length
            JS = JS/length
            DC = DC/length
            unet_score = acc
            if self.config.is_master_node:
                print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f'%(acc,SE,SP,PC,F1,JS,DC))
                
            '''
            torchvision.utils.save_image(images.data.cpu(),
                                        os.path.join(self.result_path,
                                                    '%s_valid_%d_image.png'%(self.model_type,epoch+1)))
            torchvision.utils.save_image(SR.data.cpu(),
                                        os.path.join(self.result_path,
                                                    '%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
            torchvision.utils.save_image(GT.data.cpu(),
                                         os.path.join(self.result_path,
                                                    '%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
            '''


            # Save Best U-Net model
            if unet_score > best_unet_score:
                best_unet_score = unet_score
                best_epoch = epoch
                best_unet = self.unet.state_dict()
                if self.config.is_master_node:
                    print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
                    torch.save(best_unet,unet_path)
                    
        #===================================== Test ====================================#
        del self.unet
        del best_unet
        self.build_model()
        if not self.config.distributed:
            self.unet.load_state_dict(torch.load(unet_path))

            self.unet.train(False)
            self.unet.eval()

            acc = 0.    # Accuracy
            SE = 0.     # Sensitivity (Recall)
            SP = 0.     # Specificity
            PC = 0.     # Precision
            F1 = 0.     # F1 Score
            JS = 0.     # Jaccard Similarity
            DC = 0.     # Dice Coefficient
            length=0
            for i, (images, GT) in enumerate(self.valid_loader):

                images = images.to(self.device)
                GT = GT.to(self.device)
                SR = F.sigmoid(self.unet(images))
                acc += get_accuracy(SR,GT)
                SE += get_sensitivity(SR,GT)
                SP += get_specificity(SR,GT)
                PC += get_precision(SR,GT)
                F1 += get_F1(SR,GT)
                JS += get_JS(SR,GT)
                DC += get_DC(SR,GT)

                length += images.size(0)

            acc = acc/length
            SE = SE/length
            SP = SP/length
            PC = PC/length
            F1 = F1/length
            JS = JS/length
            DC = DC/length
            unet_score = acc
            print("Test finished, acc: %.3f " % (acc))
            

            
