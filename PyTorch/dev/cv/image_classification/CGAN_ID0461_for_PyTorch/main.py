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
import numpy as np
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import apex
from apex import amp
import utils
from Generator import Generator
from Discriminator import Discriminator
import torchvision.utils as vutils


#initial steps
config_file = "config.yml"
config = utils.load_config(config_file)
utils.set_manual_seed(config)
dataset = utils.load_transformed_dataset(config)
dataloader = utils.get_dataloader(config,dataset)
################################ modify by npu ##########################################
device = torch.device("npu:0" if (torch.npu.is_available() and config['ngpu'] > 0) else "cpu")
################################ modify by npu ##########################################
print("Using device : ", device)

def train_gan(config, dataloader, device):
    #initialize models
    gen = Generator(config).to(device)
    dis = Discriminator(config).to(device)
    gen.apply(utils.init_weights)
    dis.apply(utils.init_weights)

    #setup optimizers
    gen_optimizer = apex.optimizers.NpuFusedAdam(params=gen.parameters(),
    #gen_optimizer = torch.optim.Adam(params=gen.parameters(), 
                                lr=config['lr'],
                                betas=[config['beta1'],config['beta2']])
    gen, gen_optimizer = amp.initialize(gen, gen_optimizer, opt_level="O2", loss_scale=32, combine_grad=True)
    dis_optimizer = apex.optimizers.NpuFusedAdam(params=dis.parameters(),
    #dis_optimizer = torch.optim.Adam(params=dis.parameters(), 
                                lr=config['lr'],
                                betas=[config['beta1'],config['beta2']])
    dis, dis_optimizer = amp.initialize(dis, dis_optimizer, opt_level="O2", loss_scale=32, combine_grad=True)
    #gen_optimizer = torch.optim.Adam(params=gen.parameters(), 
                                #lr=config['lr'],
                                #betas=[config['beta1'],config['beta2']])
    #dis_optimizer = torch.optim.Adam(params=dis.parameters(), 
                                #lr=config['lr'],
                                #betas=[config['beta1'],config['beta2']])

    criterion = torch.nn.BCELoss()
    fixed_latent = torch.randn(16,config['len_z'],1,1,device=device)
    dis_loss = []
    gen_loss = []
    generated_imgs = []
    iteration = 0

    #load parameters
    if(config['load_params'] and os.path.isfile("./gen_params.pth.tar")):
        print("loading params...")
        gen.load_state_dict(torch.load("./gen_params.pth.tar",map_location=torch.device(device)))
        dis.load_state_dict(torch.load("./dis_params.pth.tar",map_location=torch.device(device)))
        gen_optimizer.load_state_dict(torch.load("./gen_optimizer_state.pth.tar",map_location=torch.device(device)))
        dis_optimizer.load_state_dict(torch.load("./dis_optimizer_state.pth.tar",map_location=torch.device(device)))
        generated_imgs = torch.load("gen_imgs_array.pt",map_location=torch.device(device))
        print("loaded params.")

    #training
	################################ modify by npu ##########################################
    #start_time = time.time()
	################################ modify by npu ##########################################
    gen.train()
    dis.train()
    for epoch in range(config['epochs']):
        ################################ modify by npu ##########################################
        start_time = time.time()
        ################################ modify by npu ##########################################
        iterator = iter(dataloader)
        dataloader_flag = True
        while(dataloader_flag):
            for _ in range(config['discriminator_steps']):
                dis.zero_grad()
                gen.zero_grad()
                dis_optimizer.zero_grad()

                #sample mini-batch
                z = torch.randn(config['batch_size'],config['len_z'],1,1,device=device)

                #get images from dataloader via iterator
                try:
                    imgs, _ = next(iterator)
                    imgs = imgs.to(device)
                except:
                    dataloader_flag = False
                    break

                #compute loss
                loss_true_imgs = criterion(dis(imgs).view(-1),torch.ones(imgs.shape[0],device=device))
                with amp.scale_loss(loss_true_imgs,dis_optimizer ) as scaled_loss:
                    scaled_loss.backward()
                #loss_true_imgs.backward()
                fake_images = gen(z)    
                loss_fake_imgs = criterion(dis(fake_images.detach()).view(-1),torch.zeros(z.shape[0],device=device))
                with amp.scale_loss(loss_fake_imgs,dis_optimizer ) as scaled_loss:
                    scaled_loss.backward()
                #loss_fake_imgs.backward()

                total_error = loss_fake_imgs+loss_true_imgs
                dis_optimizer.step()
            
            #generator step
            for _ in range(config['generator_steps']):
                if(dataloader_flag==False):
                    break
                gen.zero_grad()
                dis.zero_grad()
                dis_optimizer.zero_grad()
                gen_optimizer.zero_grad()

                #z = torch.randn(config['batch_size'],config['len_z'])   #sample mini-batch
                loss_gen = criterion(dis(fake_images).view(-1),torch.ones(z.shape[0],device=device))    #compute loss
    
                #update params
                with amp.scale_loss(loss_gen,gen_optimizer) as scaled_loss:
                    scaled_loss.backward()
                #loss_gen.backward()
                gen_optimizer.step()

            iteration+=1
            
            #log and save variable, losses and generated images
            if(iteration%100)==0:
                elapsed_time = time.time()-start_time
                ################################ modify by npu ##########################################
                start_time = time.time()
                ################################ modify by npu ##########################################
                dis_loss.append(total_error.mean().item())
                gen_loss.append(loss_gen.mean().item())

                with torch.no_grad():
                    generated_imgs.append(gen(fixed_latent).detach())    #generate image
                    torch.save(generated_imgs,"gen_imgs_array.pt")

                print("Iteration:%d, Dis Loss:%.4f, Gen Loss:%.4f, time elapsed:%.4f"%(iteration,dis_loss[-1],gen_loss[-1],elapsed_time))
                
                if( config['save_params'] and iteration%400==0):
                    print("saving params...")
                    torch.save(gen.state_dict(), "./gen_params.pth.tar")
                    torch.save(dis.state_dict(), "./dis_params.pth.tar")
                    torch.save(dis_optimizer.state_dict(), "./dis_optimizer_state.pth.tar")
                    torch.save(gen_optimizer.state_dict(), "./gen_optimizer_state.pth.tar")
                    print("saved params.")

    #plot errors
    utils.save_loss_plot(gen_loss,dis_loss)

    #plot generated images
    #utils.save_result_images(next(iter(dataloader))[0][:15].to(device),generated_imgs[-1],4,config)
    utils.save_result_images(next(iter(dataloader))[0][:15].cpu(),generated_imgs[-1].cpu(),4,config)
    
    #save generated images so see what happened
    torch.save(generated_imgs,"gen_imgs_array.pt")

    #save gif
    utils.save_gif(generated_imgs,4,config)

def generate_images(config, dataloader, device):
    gen = Generator(config).to(device)
    if(config['load_params'] and os.path.isfile("./gen_params.pth.tar")):
        print("loading params...")
        gen.load_state_dict(torch.load("./gen_params.pth.tar",map_location=torch.device(device)))
        print("loaded params...")
    gen.eval()
    z=torch.randn(16,config['len_z'],1,1,device=device)
    with torch.no_grad():
        fake_images=gen(z)  
    utils.save_result_images(next(iter(dataloader))[0][:16].to(device),fake_images.cpu(),4,config)

train_gan(config, dataloader, device)