# -*- coding: utf-8 -*- 
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
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.npu
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import DnCNN
from dataset import Traindata, Evldata
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import batch_PSNR, weights_init_kaiming, AverageMeter
from apex import amp        # apex Add
import apex 
import torch.multiprocessing as mp
import time


parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=str, default="F", help="write T or Ture means create h5py dataset")
parser.add_argument('--data_path', type=str, help='path of dataset')
parser.add_argument("--batchSize", type=int, default=512, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default=".", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
parser.add_argument('--gpu_use_num', default=8, type=int, help='how many gpus you want use')
parser.add_argument('--device-list', default='0,1,2,3,4,5,6,7', type=str, help='device id list')
parser.add_argument('--worldsize', default=8, type=int, help='number of nodes for distributed training')
opt = parser.parse_args()


def getDiv(epo):
    """ lr div value get """
    if epo < 15:
        return 1.
    if epo < 25:
        return 2.
    if epo < 30:
        return 3.
    if epo < 60:
        return 4.
    return 1.


def device_id_to_process_device_map(device_list):
    """ find device """
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()
    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id
    return process_device_map


def main():
    """ 8p in"""
    opt.process_device_map = device_id_to_process_device_map(opt.device_list)
    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=opt.gpu_use_num, args=(ngpus_per_node, opt))  #8卡 给 8进程


def main_worker(gpu, gpu_nums, opt):
    """ spawn thread """
    opt.gpu = opt.process_device_map[gpu]
    print ("gpu -> ", gpu)
    print ("all :", gpu_nums)
    #分P初始化
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29688'
    os.environ['WORLD_SIZE'] = '8'
    dist.init_process_group(backend="hccl", init_method='env://', world_size=opt.worldsize, rank=gpu) 
   
    #切分数据集 并加载
    dataset_train = Traindata(data_path=opt.data_path, getDataSet='train')
    dataset_val = Evldata(data_path=opt.data_path, getDataSet='Set68')
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)  
    loader_train = DataLoader(dataset=dataset_train, batch_size=opt.batchSize, \
        pin_memory=True, shuffle=(train_sampler is None), sampler=train_sampler, drop_last=True)
    
    print("# of training samples: %d" % int(len(dataset_train)))
    
    #npu的device确定
    print("use gpu num", opt.gpu)
    loc = 'npu:{}'.format(opt.gpu)
    torch.npu.set_device(loc)
    
    # 创建网络
    net = DnCNN(channels = 1, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    net=net.to(loc)       
    
    #opt.lr = 0.008
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    criterion = nn.MSELoss(reduction='sum')
    net, optimizer = amp.initialize(net, optimizer, opt_level = "O2", loss_scale = 128.0) #              
    model = DDP(net, device_ids=[gpu])       
    criterion.to(loc) 
    
    #时间记录申明
    step_time = AverageMeter('Time', ':6.3f')
    step_start=time.time()
    data_time = AverageMeter('Time', ':6.3f')
    data_start=time.time()
       
    maxPsnr=0
    for epoch in range(opt.epochs): #6000/8
        train_sampler.set_epoch(epoch)  
        current_lr = opt.lr / getDiv(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        #开始训练
        model.train()
        for i, dataLoad in enumerate(loader_train, 0):
            data_difTime=time.time() - data_start
            data_time.update(data_difTime)      
            step_start=time.time()
            
            imgn_train, img_train, noise = dataLoad
            img_train, imgn_train = img_train.to(loc), imgn_train.to(loc)
            noise = noise.to(loc)
            out_train = model(imgn_train)
            loss = criterion(out_train, noise) / (imgn_train.size()[0] * 2)
                       
            with amp.scale_loss(loss, optimizer) as scaled_loss: 
                scaled_loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            torch.npu.synchronize()         #同步时间
            step_difTime=time.time() - step_start
            step_time.update(step_difTime)
            data_start = time.time()       
            if i % 40 == 2 and int(time.time()) % 8 == opt.gpu: 
                print('epoch[{}] load[{}/{}] loss: {:.3f} '.format(epoch, i + 1, len(loader_train), loss.item()))                    
                print('dataTime[{:.3f}]/[{:.3f}]  stepTime[{:.3f}/{:.3f}]  FPS: {:.3f}'.format(data_difTime, \
                    data_time.avg, step_difTime, step_time.avg, opt.batchSize * opt.worldsize / (data_difTime + step_difTime)))
                
       ## 一批次训练结束，固定模型
        model.eval()
        psnr_val = 0
        for k in range(len(dataset_val)):
            noiseData, clearData, noise = dataset_val[k]
            img_val = torch.unsqueeze(clearData, 0)
            imgn_val = torch.unsqueeze(noiseData, 0)
            
            img_val, imgn_val = img_val.to(loc), imgn_val.to(loc)  
            
            out_val = torch.clamp(imgn_val - model(imgn_val), 0., 1.)
            psnr_val += batch_PSNR(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
       
        print("\n[epoch %d] [gpu %d] PSNR_val: %.4f" % (epoch, opt.gpu, psnr_val))
    
        if opt.gpu == 0 and psnr_val > maxPsnr:
            maxPsnr = psnr_val
            print("save and maxPsnr = ", maxPsnr)
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net8p.pth'))
    if opt.gpu == 0:
        print("gpuNum: ", opt.gpu, "gpu num finnal Psnr:", maxPsnr)
        
if __name__ == "__main__":
    main()




