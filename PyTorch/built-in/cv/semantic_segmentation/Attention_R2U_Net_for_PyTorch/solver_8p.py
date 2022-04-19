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
import torch.distributed as dist 
from data_loader import get_dist_loader, get_loader,ImageFolder
from apex import amp 
import apex

def train_8p(rank, npus, config):
    rank = rank
    dist.init_process_group(backend=config.dist_backend, world_size=config.npus, rank=rank)
    torch.npu.set_device(rank)
    model_unet = R2AttU_Net(img_ch=3, output_ch=1,t=config.t)
    model_unet = model_unet.to("npu")
    # optimizer = optim.Adam(list(model_unet.parameters()), config.lr, [config.beta1, config.beta2])
    optimizer = apex.optimizers.NpuFusedAdam(list(model_unet.parameters()), config.lr, [config.beta1, config.beta2])
    
    if config.use_apex:
        model_unet, optimizer = amp.initialize(model_unet, optimizer, 
                                    opt_level=config.apex_level,loss_scale=config.loss_scale, combine_grad=True)
    model_unet = torch.nn.parallel.DistributedDataParallel(model_unet, device_ids=[rank], 
                                                           broadcast_buffers=False)
    
    train_loader = get_dist_loader(image_path=config.train_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='train',
                            augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(image_path=config.valid_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='valid',
                            augmentation_prob=0.)
    criterion = torch.nn.BCELoss()
    
    lr = config.lr
    best_unet_score = 0.
    unet_path = os.path.join(config.result_path, '%d-%s-%d-%.4f-%d-%.4f.pkl' %(rank, 
                config.model_type,config.num_epochs,config.lr,config.num_epochs_decay,config.augmentation_prob))

    for epoch in range(config.num_epochs):
        model_unet.train(True)
        epoch_loss = 0. 
        acc = 0.	# Accuracy
        length = 0
        threshold = 0.5
        steps = len(train_loader)
        for i, (images, GT) in enumerate(train_loader):
            # GT : Ground Truth
            images = images.to("npu")
            GT = GT.to("npu")
            if i == 10:
                start_time = time.time()
            step_start_time = time.time()
            # SR : Segmentation Result
            SR = model_unet(images)
            SR_probs = F.sigmoid(SR)
            SR_flat = SR_probs.view(SR_probs.size(0),-1)

            GT_flat = GT.view(GT.size(0),-1)
            loss = criterion(SR_flat,GT_flat)
            epoch_loss += loss.item()

            # Backprop + optimize
            model_unet.zero_grad()
            if config.use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            if rank == 0:
                print('Epoch [%d/%d], Step: %d, FPS: %.2f' % (
                        epoch+1, config.num_epochs, i, config.npus*config.batch_size/(time.time() - step_start_time)))
            SR_ac = SR > threshold
            GT_ac = GT == torch.max(GT)
            acc += get_accuracy(SR_ac, GT_ac)
            length += 1

        acc = acc/length

        # Print the log info
        if rank == 0:
            print('Rank %d , Epoch [%d/%d], Loss: %.4f, [Training] Acc: %.4f,  FPS: %.2f' % (
                rank, epoch+1, config.num_epochs, epoch_loss, acc , config.batch_size*(steps-10)/(time.time() - start_time)))
        # Decay learning rate
        if (epoch+1) % 10 == 0:
            lr = lr/2.
            # lr -= (config.lr / float(config.num_epochs_decay))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print ('Decay learning rate to lr: {}.'.format(lr))
        
        
        #===================================== Validation ====================================#
        if rank == 0:
            model_unet.eval()

            acc = 0.	# Accuracy
            length=0
            for i, (images, GT) in enumerate(valid_loader):

                images = images.to("npu")
                GT = GT.to("npu")
                SR = F.sigmoid(model_unet(images))
                SR_ac = SR > threshold
                GT_ac = GT == torch.max(GT)
                acc += get_accuracy(SR_ac, GT_ac)
                    
                length += 1
                
            acc = acc/length

            unet_score = acc#JS + DC

            print('[Validation] Rank: %d, Epoch %d,Acc: %.4f'%(rank, epoch, acc))
            
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
                best_unet = model_unet.state_dict()
                print('Best %s model score : %.4f'%(config.model_type,best_unet_score))
                torch.save(best_unet,unet_path)
                print("Validation Best", [config.model_type,acc,config.lr,best_epoch,\
                    config.num_epochs,config.num_epochs_decay,config.augmentation_prob])
            