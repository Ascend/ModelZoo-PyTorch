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
# import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
from apex import amp
import apex

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
        self.use_apex = config.use_apex
        self.apex_level = config.apex_level
        self.loss_scale = config.loss_scale

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.result_path = config.result_path
        self.mode = config.mode

        self.device = torch.device('npu' if torch.npu.is_available() else 'cpu')
        self.model_type = config.model_type
        self.test_model_path = config.test_model_path
        self.t = config.t
        self.pretrain = config.pretrain
        self.pretrain_path = config.pretrain_path 
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type =='U_Net':
            self.unet = U_Net(img_ch=3,output_ch=1)
        elif self.model_type =='R2U_Net':
            self.unet = R2U_Net(img_ch=3,output_ch=1,t=self.t)
        elif self.model_type =='AttU_Net':
            self.unet = AttU_Net(img_ch=3,output_ch=1)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=3,output_ch=1,t=self.t)
        
        if self.pretrain:
            self.unet.load_state_dict(torch.load(self.pretrain_path, map_location="cpu"), strict=False)

        self.unet.to(self.device)
        if self.mode == "test":
            return
        # self.optimizer = optim.Adam(list(self.unet.parameters()),
        #                               self.lr, [self.beta1, self.beta2])
        self.optimizer = apex.optimizers.NpuFusedAdam(list(self.unet.parameters()),
                                      self.lr, [self.beta1, self.beta2])
        if self.use_apex:
            self.unet, self.optimizer = amp.initialize(self.unet, self.optimizer, 
                                        opt_level=self.apex_level,loss_scale=self.loss_scale, combine_grad=True)

        # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
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
        
        unet_path = os.path.join(self.result_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))

        # U-Net Train
        # Train for Encoder
        lr = self.lr
        best_unet_score = 0.
        
        for epoch in range(self.num_epochs):

            self.unet.train(True)
            epoch_loss = 0
            
            acc = 0.	# Accuracy
            SE = 0.		# Sensitivity (Recall)
            SP = 0.		# Specificity
            PC = 0. 	# Precision
            F1 = 0.		# F1 Score
            JS = 0.		# Jaccard Similarity
            DC = 0.		# Dice Coefficient
            length = 0
            threshold = 0.5
            steps = len(self.train_loader)
            for i, (images, GT) in enumerate(self.train_loader):
                # GT : Ground Truth
                if i > 10:
                    start_time = time.time()
                images = images.to(self.device)
                GT = GT.to(self.device)

                # SR : Segmentation Result
                SR = self.unet(images)
                SR_probs = F.sigmoid(SR)
                SR_flat = SR_probs.view(SR_probs.size(0),-1)

                GT_flat = GT.view(GT.size(0),-1)
                loss = self.criterion(SR_flat,GT_flat)
                epoch_loss += loss.item()

                # Backprop + optimize
                self.reset_grad()
                if self.use_apex:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                self.optimizer.step()

                SR_ac = SR > threshold
                GT_ac = GT == torch.max(GT)
                acc += get_accuracy(SR_ac, GT_ac)
                SE += get_sensitivity(SR_ac, GT_ac)
                SP += get_specificity(SR_ac, GT_ac)
                PC += get_precision(SR_ac, GT_ac)
                F1 += get_F1(SR_ac, GT_ac)
                JS += get_JS(SR_ac, GT_ac)
                DC += get_DC(SR_ac, GT_ac)
                length += 1

            acc = acc/length
            SE = SE/length
            SP = SP/length
            PC = PC/length
            F1 = F1/length
            JS = JS/length
            DC = DC/length

            # Print the log info
            print('Epoch [%d/%d], Loss: %.4f, [Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, FPS: %.3f' % (
                    epoch+1, self.num_epochs, \
                    epoch_loss,acc,SE,SP,PC,F1,JS,DC, self.batch_size*(steps-10)/(time.time() - start_time)))

        

            # Decay learning rate
            if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
                lr -= (self.lr / float(self.num_epochs_decay))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print ('Decay learning rate to lr: {}.'.format(lr))
            
            
            #===================================== Validation ====================================#
            self.unet.eval()

            acc = 0.	# Accuracy
            SE = 0.		# Sensitivity (Recall)
            SP = 0.		# Specificity
            PC = 0. 	# Precision
            F1 = 0.		# F1 Score
            JS = 0.		# Jaccard Similarity
            DC = 0.		# Dice Coefficient
            length=0
            for i, (images, GT) in enumerate(self.valid_loader):

                images = images.to(self.device)
                GT = GT.to(self.device)
                SR = F.sigmoid(self.unet(images))

                SR_ac = SR > threshold
                GT_ac = GT == torch.max(GT)
                acc += get_accuracy(SR_ac, GT_ac)
                SE += get_sensitivity(SR_ac, GT_ac)
                SP += get_specificity(SR_ac, GT_ac)
                PC += get_precision(SR_ac, GT_ac)
                F1 += get_F1(SR_ac, GT_ac)
                JS += get_JS(SR_ac, GT_ac)
                DC += get_DC(SR_ac, GT_ac)
                    
                length += 1
                
            acc = acc/length
            SE = SE/length
            SP = SP/length
            PC = PC/length
            F1 = F1/length
            JS = JS/length
            DC = DC/length
            unet_score = JS + DC

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
                print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
                torch.save(best_unet,unet_path)

                print("Validation Best ", [self.model_type,acc,SE,SP,PC,F1,JS,DC,self.lr,best_epoch,self.num_epochs,self.num_epochs_decay,self.augmentation_prob])
        
            #===================================== Test ====================================#
    def test(self):
        threshold = 0.5
        pre_dict = torch.load(self.test_model_path)
        new_dict = {}
        if list(pre_dict.keys())[0].startswith("module"):
            for key, value in pre_dict.items():
                name = key[7:]
                new_dict[name] = value
        else:
            new_dict = pre_dict
        self.unet.load_state_dict(new_dict)
        self.unet.eval()

        acc = 0.	# Accuracy
        SE = 0.		# Sensitivity (Recall)
        SP = 0.		# Specificity
        PC = 0. 	# Precision
        F1 = 0.		# F1 Score
        JS = 0.		# Jaccard Similarity
        DC = 0.		# Dice Coefficient
        length=0
        for i, (images, GT) in enumerate(self.test_loader):

            images = images.to(self.device)
            GT = GT.to(self.device)
            SR = F.sigmoid(self.unet(images))
            SR_ac = SR > threshold
            GT_ac = GT == torch.max(GT)
            acc += get_accuracy(SR_ac, GT_ac)
            SE += get_sensitivity(SR_ac, GT_ac)
            SP += get_specificity(SR_ac, GT_ac)
            PC += get_precision(SR_ac, GT_ac)
            F1 += get_F1(SR_ac, GT_ac)
            JS += get_JS(SR_ac, GT_ac)
            DC += get_DC(SR_ac, GT_ac)
                    
            length += 1
                
        acc = acc/length
        SE = SE/length
        SP = SP/length
        PC = PC/length
        F1 = F1/length
        JS = JS/length
        DC = DC/length
        unet_score = JS + DC
        print("Test finished, model checkpoint name:",self.test_model_path, " and acc: %.3f " % (acc))




        
