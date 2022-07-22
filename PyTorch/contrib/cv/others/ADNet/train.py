# coding=utf-8
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import apex.amp as amp
import apex
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torch.nn.modules.loss import _Loss 
from models import ADNet
from dataset import prepare_data, Dataset
from collections import OrderedDict
from utils import *
if torch.__version__ >= '1.8':
    import torch_npu

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--resume", type=bool, default=False, help="resume training from .pth")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs")
parser.add_argument("--logdir", type=str, default="", help='path of log files')
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files?path to .pth')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=15, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=15, help='noise level used on validation set')
parser.add_argument("--is_distributed", type=int, default=0, help='choose ddp or not')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--DeviceID', type=str, default="0")
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--world_size", default=-1, type=int)
parser.add_argument("--loss_scale", default=128, type=int)
'''
parser.add_argument("--clip",type=float,default=0.005,help='Clipping Gradients. Default=0.4') #tcw201809131446tcw
parser.add_argument("--momentum",default=0.9,type='float',help = 'Momentum, Default:0.9') #tcw201809131447tcw
parser.add_argument("--weight-decay","-wd",default=1e-3,type=float,help='Weight decay, Default:1e-4') #tcw20180913347tcw
'''
opt = parser.parse_args()

def proc_nodes_module(checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if "module." in k:
            name = k.replace("module.", "")
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)
    def forward(self, input, target):
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)

def main():
    # Load dataset
    t1 = time.time()
    #distributed training judgement
    if opt.is_distributed == 0:
        local_device = torch.device(f'npu:{opt.DeviceID}')
        torch.npu.set_device(local_device)
        print("using npu :{}".format(opt.DeviceID))
    else:
        os.environ['MASTER_ADDR'] = '127.0.0.1'         #can change to real ip
        os.environ['MASTER_PORT'] = '29688'         #set port can be change
        os.environ['RANK'] = str(opt.local_rank)
        local_device = torch.device(f'npu:{opt.local_rank}')
        torch.npu.set_device(local_device)
        if opt.local_rank == 0:
            print("using npu :{}".format(opt.DeviceID))
        dist.init_process_group(backend='hccl',world_size=opt.world_size, rank=opt.local_rank)

    #set direction for saving model 
    save_dir = opt.outf + 'sigma' + str(opt.noiseL) + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '/'
    # save_dir = opt.outf + '_' + str(opt.noiseL) + '_'+str(opt.num_gpus) + 'full' + '_' + 'lossscale8'

    #create direction
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Load dataset
    if opt.local_rank == 0:
        print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    if opt.is_distributed == 0:
        loader_train = DataLoader(dataset=dataset_train, num_workers=16, batch_size=opt.batchSize, shuffle=True, drop_last=True)
    else:
        train_sampler = DistributedSampler(dataset_train)
        loader_train = DataLoader(dataset=dataset_train, sampler=train_sampler, num_workers=16,
                                  batch_size=opt.batchSize, pin_memory=False, drop_last=True)
    if opt.local_rank == 0:
         print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
    net = ADNet(channels=1, num_of_layers=opt.num_of_layers)
    model = net.to(local_device)
    # Optimizer选择
    # optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # optimizer = optim.RMSProp(model.parameters(), lr=opt.lr,alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    # optimizer = apex.optimizers.NpuFusedAdamW(model.parameters(), lr=opt.lr, betas=(0.9,0.999))
    # optimizer = optim.Adam(model.parameters(), lr=opt.lr,eps=1e-08)
    optimizer = apex.optimizers.NpuFusedSGD(model.parameters(), lr=opt.lr, momentum=0.9, nesterov=True)
    #load pretrained model or initialize model 
    if opt.resume == True:
        path_checkpoint = os.path.join(save_dir, 'best_model.pth')
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dic'])
        start_epoch = checkpoint['epoch']+1
        #checkpoint = proc_nodes_module(checkpoint)
        #model.load_state_dict(checkpoint['model_state_dict'])
        # checkpoint = torch.load(os.path.join(opt.logdir, 'best_model.pth'), map_location=local_device)
        # checkpoint = proc_nodes_module(checkpoint)
        # model.load_state_dict(checkpoint)
        model = model.npu()
    else:
        # set model and optimizer according to opt_level，opt_level can choose O2 or O1
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=opt.loss_scale)
        start_epoch = 0
    criterion = nn.MSELoss(reduction='sum')
    criterion.cpu()
    if opt.is_distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], broadcast_buffers=False)
    noiseL_B=[0,55]         # ingnored when opt.mode=='S'
    psnr_list = [] 
    #noise = torch.FloatTensor(torch.Size([128,1,50,50])).normal_(mean=0, std=opt.noiseL/255.)
    #noise.npu()

    #start training
    for epoch in range(start_epoch, opt.epochs):
        if opt.is_distributed == 1:
            train_sampler.set_epoch(epoch)
        if epoch <= opt.milestone:
            current_lr = opt.lr
        if epoch > 30 and  epoch <=60:
            current_lr  =  opt.lr/1.
        if epoch > 60  and epoch <=90:
            # current_lr = opt.lr/100.
            current_lr = opt.lr /100.
        if epoch >90 and epoch <=120:
            current_lr = opt.lr/1000.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        if opt.local_rank == 0:
            print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            if i == 2:
                time_step2 = time.time()
            model.train()
            img_train = data
            if opt.mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
            if opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0,:,:,:].size()
                    noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.) 
            if opt.is_distributed == 0 and epoch == 0 and i == 6:
                with torch.autograd.profiler.profile(use_npu=True) as prof:
                    imgn_train = img_train + noise
                    img_train, imgn_train = Variable(img_train.npu()), Variable(imgn_train.npu())
                    noise = Variable(noise.npu())
                    out_train = model(imgn_train)
                    loss = (criterion(out_train.cpu(), img_train.cpu()) / (imgn_train.size()[0] * 2)).npu()
                    optimizer.zero_grad()
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    optimizer.step()
                prof.export_chrome_trace("output.prof")         # "output.prof"
            else:
                imgn_train = img_train + noise
                img_train, imgn_train = Variable(img_train.npu()), Variable(imgn_train.npu())
                noise = Variable(noise.npu())
                out_train = model(imgn_train)
                loss = (criterion(out_train.cpu(), img_train.cpu()) / (imgn_train.size()[0] * 2)).npu()
                optimizer.zero_grad()
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
            #eval
            model.eval()
            out_train = torch.clamp(model(imgn_train), 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            if opt.local_rank == 0:
                if (i + 1) == len(loader_train):
                    time_avg = time.time() - time_step2
                    fps = opt.num_gpus * opt.batchSize * len(loader_train) / time_avg
                    print("[epoch %d][%d/%d] fps: %.4f time_avg: %.4f" %
                          (epoch + 1, i + 1, len(loader_train), fps, time_avg))
                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f scaled_loss: %.4f " %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train, scaled_loss.item()))
        model.eval()

        # computing PSNR
        psnr_val = 0
        best_psnr = 0
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            torch.manual_seed(0) #set the seed 
            noisy = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
            imgn_val = img_val + noisy
            img_val, imgn_val = Variable(img_val.npu()), Variable(imgn_val.npu(),requires_grad=False)
            out_val = torch.clamp(model(imgn_val), 0., 1.)
            stream = torch.npu.current_stream()
            stream.synchronize()
            psnr_val += batch_PSNR(out_val, img_val, 1.)
            stream = torch.npu.current_stream()
            stream.synchronize()
        psnr_val /= len(dataset_val)
        psnr_val1 = str(psnr_val) 
        psnr_list.append(psnr_val1) 
        if opt.local_rank == 0:
            print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        #set model name
        model_name = 'model'+ '_' + str(opt.resume)+ '_' + str(epoch+1) + '.pth'
        #save checkpoint
        checkpoint = {"model_state_dict": net.state_dict(),
                      "optimizer_state_dic": optimizer.state_dict(),
                      "loss": loss,
                      "epoch": epoch}
        torch.save(checkpoint, os.path.join(save_dir, model_name))
        # torch.save(model.state_dict(), os.path.join(save_dir, model_name))
        if best_psnr < psnr_val:
            # torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))    #.pth
            checkpoint = {"model_state_dict": net.state_dict(),
                          "optimizer_state_dic": optimizer.state_dict(),
                          "loss": loss,
                          "epoch": epoch}
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
    filename = save_dir + 'psnr.txt'            #保存训练过程中的验证集PSNR
    f = open(filename,'w') 
    for line in psnr_list:  
        f.write(line+'\n') 
    f.close()
    t2 = time.time()
    t = t2-t1
    if opt.local_rank == 0:
        print ("total training used time:",t) 

if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=50, stride=40, aug_times=1) 
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
    main()
