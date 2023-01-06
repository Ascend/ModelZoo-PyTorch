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
if torch.__version__ >= "1.8":
    import torch_npu
import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
from metrics import psnr,ssim
from models import *
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
#from tensorboardX import SummaryWriter
import torchvision.utils as vutils
warnings.filterwarnings('ignore')
from option import opt,model_name,log_dir
from data_utils import *
from torchvision.models import vgg16
import torch.npu
import os
try:
    from apex import amp
except ImportError:
    amp=None

#替换亲和性接口
import apex
#替换亲和性接口


NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')
print("========device_id:", NPU_CALCULATE_DEVICE)
print('log_dir :',log_dir)
print('model_name:',model_name)

models_={
    'ffa':FFA(gps=opt.gps,blocks=opt.blocks),
}
loaders_={
    'its_train':ITS_train_loader,
    'its_test':ITS_test_loader,
    'ots_train':OTS_train_loader,
    'ots_test':OTS_test_loader
}
start_time=time.time()
T=opt.steps	
def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
    lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
    return lr

def train(net,loader_train,loader_test,optim,criterion):
    losses=[]
    start_step=0
    max_ssim=0
    max_psnr=0
    ssims=[]
    psnrs=[]
    if opt.resume and os.path.exists(opt.model_dir):
        print(f'resume from {opt.model_dir}')
        ckp=torch.load(opt.model_dir,map_location='cpu')
        losses=ckp['losses']
        net.load_state_dict(ckp['model'],False)
        start_step=ckp['step']
        max_ssim=ckp['max_ssim']
        max_psnr=ckp['max_psnr']
        psnrs=ckp['psnrs']
        ssims=ckp['ssims']
        print(f'start_step:{start_step} start training ---')
    else :
        print('train from scratch *** ')
    for step in range(start_step+1,opt.steps+1):

        begin_time = time.time()
        net.train()
        lr=opt.lr
        if not opt.no_lr_sche:
            lr=lr_schedule_cosdecay(step,T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr
        x,y=next(iter(loader_train))
        x=x.to(f'npu:{NPU_CALCULATE_DEVICE}');y=y.to(f'npu:{NPU_CALCULATE_DEVICE}')
        out=net(x)
        loss=criterion[0](out,y)
        if opt.perloss:
            loss2=criterion[1](out,y)
            loss=loss+0.04*loss2
        if opt.apex:
            with amp.scale_loss(loss,optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optim.step()
        optim.zero_grad()
        losses.append(loss.item())
        fps = opt.bs / (time.time() - begin_time)
        if step < 3:
            print("step_time = {:.4f}".format(time.time() - begin_time), flush=True)
        print(f'\rtrain loss :{loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time()-start_time)/60 :.1f}|fps :{fps:.4f}',end='',flush=True)

        #with SummaryWriter(logdir=log_dir,comment=log_dir) as writer:
        #	writer.add_scalar('data/loss',loss,step)

        '''
        # 导出PT profiler
        with torch.autograd.profiler.profile(use_npu=True) as prof:
            begin_time = time.time()
            net.train()
            lr = opt.lr
            if not opt.no_lr_sche:
                lr = lr_schedule_cosdecay(step, T)
                for param_group in optim.param_groups:
                    param_group["lr"] = lr
            x, y = next(iter(loader_train))
            x = x.to(f'npu:{NPU_CALCULATE_DEVICE}');
            y = y.to(f'npu:{NPU_CALCULATE_DEVICE}')
            out = net(x)
            loss = criterion[0](out, y)
            if opt.perloss:
                loss2 = criterion[1](out, y)
                loss = loss + 0.04 * loss2
            if opt.apex:
                with amp.scale_loss(loss, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optim.step()
            optim.zero_grad()
            losses.append(loss.item())
            fps = opt.bs / (time.time() - begin_time)
            print(
                f'\rtrain loss :{loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time() - start_time) / 60 :.1f}|fps :{fps:.4f}',
                end='', flush=True)

            # with SummaryWriter(logdir=log_dir,comment=log_dir) as writer:
            #	writer.add_scalar('data/loss',loss,step)
        prof.export_chrome_trace("./npu_profiler_iter{}.json".format(step))
        # 导出PT profiler
        '''
        if step % opt.eval_step ==0 :
            with torch.no_grad():
                ssim_eval,psnr_eval=test(net,loader_test, max_psnr,max_ssim,step)

            print(f'\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

            # with SummaryWriter(logdir=log_dir,comment=log_dir) as writer:
            # 	writer.add_scalar('data/ssim',ssim_eval,step)
            # 	writer.add_scalar('data/psnr',psnr_eval,step)
            # 	writer.add_scalars('group',{
            # 		'ssim':ssim_eval,
            # 		'psnr':psnr_eval,
            # 		'loss':loss
            # 	},step)
            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            if ssim_eval > max_ssim and psnr_eval > max_psnr :
                max_ssim=max(max_ssim,ssim_eval)
                max_psnr=max(max_psnr,psnr_eval)
                torch.save({
                            'step':step,
                            'max_psnr':max_psnr,
                            'max_ssim':max_ssim,
                            'ssims':ssims,
                            'psnrs':psnrs,
                            'losses':losses,
                            'model':net.state_dict()
                },opt.model_dir)
                print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

    np.save(f'./numpy_files/{model_name}_{opt.steps}_losses.npy',losses)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_ssims.npy',ssims)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_psnrs.npy',psnrs)

def test(net,loader_test,max_psnr,max_ssim,step):
    net.eval()
    torch.npu.empty_cache()
    ssims=[]
    psnrs=[]
    #s=True
    for i ,(inputs,targets) in enumerate(loader_test):
        inputs=inputs.to(f'npu:{NPU_CALCULATE_DEVICE}');targets=targets.to(f'npu:{NPU_CALCULATE_DEVICE}')
        pred=net(inputs)
        # # print(pred)
        # tfs.ToPILImage()(torch.squeeze(targets.cpu())).save('111.png')
        # vutils.save_image(targets.cpu(),'target.png')
        # vutils.save_image(pred.cpu(),'pred.png')
        ssim1=ssim(pred,targets).item()
        psnr1=psnr(pred,targets)
        ssims.append(ssim1)
        psnrs.append(psnr1)
        #if (psnr1>max_psnr or ssim1 > max_ssim) and s :
        #		ts=vutils.make_grid([torch.squeeze(inputs.cpu()),torch.squeeze(targets.cpu()),torch.squeeze(pred.clamp(0,1).cpu())])
        #		vutils.save_image(ts,f'samples/{model_name}/{step}_{psnr1:.4}_{ssim1:.4}.png')
        #		s=False
    return np.mean(ssims) ,np.mean(psnrs)


if __name__ == "__main__":
    loader_train=loaders_[opt.trainset]
    loader_test=loaders_[opt.testset]
    net=models_[opt.net]
    net=net.to(f'npu:{NPU_CALCULATE_DEVICE}')
    if opt.device=='npu':
        net=net
        cudnn.benchmark=True
    criterion = []
    criterion.append(nn.L1Loss().to(f'npu:{NPU_CALCULATE_DEVICE}'))
    if opt.perloss:
            vgg_model = vgg16(pretrained=True).features[:16]
            vgg_model = vgg_model.to(f'npu:{NPU_CALCULATE_DEVICE}')
            for param in vgg_model.parameters():
                param.requires_grad = False
            criterion.append(PerLoss(vgg_model).to(f'npu:{NPU_CALCULATE_DEVICE}'))
    # 替换亲和性接口
    # optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()),lr=opt.lr, betas = (0.9, 0.999), eps=1e-08)
    optimizer = apex.optimizers.NpuFusedAdam(params=filter(lambda x: x.requires_grad, net.parameters()),lr=opt.lr, betas = (0.9, 0.999), eps=1e-08)
    # 替换亲和性接口
    if opt.apex:
        net, optimizer = amp.initialize(net, optimizer, opt_level=opt.apex_opt_level, combine_grad=True)
    optimizer.zero_grad()
    train(net,loader_train,loader_test,optimizer,criterion)


