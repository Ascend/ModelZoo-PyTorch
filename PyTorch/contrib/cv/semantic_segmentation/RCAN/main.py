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

import argparse
import os
import torch
from torch import nn
import torch.multiprocessing as mp
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from model import RCAN
from dataset import Dataset,Dataset_test_label
from utils import AverageMeter,device_id_to_process_device_map,information_print,timer
import shutil
import matplotlib.pyplot as plt
import numpy as np 
import time 
import PIL.Image as pil_image
import skimage.io as io
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
from apex import amp
import apex
import sys


parser = argparse.ArgumentParser()
# train and test setting
parser.add_argument('--arch', type=str, default='RCAN')
parser.add_argument('--train_dataset_dir', type=str, required=True)
parser.add_argument('--test_dataset_dir', type=str, required=True)
parser.add_argument('--outputs_dir', type=str, required=True)
parser.add_argument('--patch_size', type=int, default=48)
parser.add_argument('--batch_size', type=int, default=160)
parser.add_argument('--num_epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--seed', type=int, default=123)

# model setting
parser.add_argument('--scale', type=int, default=2)
parser.add_argument('--num_features', type=int, default=64)
parser.add_argument('--num_rg', type=int, default=10)
parser.add_argument('--num_rcab', type=int, default=20)
parser.add_argument('--reduction', type=int, default=16)

# continue and iffinetuning setting
parser.add_argument('--ifcontinue',action='store_true', default=False, help='if continue to train the model')
parser.add_argument('--checkpoint_path', type=str,default="model_best_amp.pth", help='the path of checkpoint to load')
parser.add_argument('--iffinetuning',action='store_true', default=False, help='if iffinetuning to train the model')
parser.add_argument('--finetuning_checkpoint_path', type=str,default="", help='the path of checkpoint to load')

# amp setting
parser.add_argument('--amp', default=False, action='store_true', help='if use amp to train the model')
parser.add_argument('--loss_scale', default=128.0, type=float, help='amp setting: loss scale, default 128.0')
parser.add_argument('--opt_level', default='O2', type=str, help='amp setting: opt level, default O2')

# device and process setting
parser.add_argument('--device', type=str,default= "gpu", help='npu or gpu')
parser.add_argument('--device_list', type=str,default= '0,1,2,3,4,5,6,7')
parser.add_argument('--device_id', type=int, default=None,help='index of gpu/npu to use')
parser.add_argument('--world_size', type=int, default=1,help='number of nodes for distributed training')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
def flush_print(func):
    def new_print(*args, **kwargs):
        func(*args, **kwargs)
        sys.stdout.flush()
    return new_print
print = flush_print(print)               

def save_checkpoint(opt,epoch,checkpoint_performance,checkpoint_time,model,optimizer):
    # checkpoint_path = os.path.join(opt.outputs_dir,'{}_epoch{}_last{}.pth'.format(opt.arch,str(epoch).zfill(4), "_amp" if opt.amp else ""))
    checkpoint_path = os.path.join(opt.outputs_dir,'{}_epoch_last{}.pth'.format(opt.arch, "_amp" if opt.amp else ""))
    checkpoint = {
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'epoch':epoch,
            'amp':amp.state_dict() if opt.amp else None,
            'best':np.array(checkpoint_performance)[:,0].max(),
            'ck_performance':checkpoint_performance,
            'ck_time':checkpoint_time
        }
    torch.save(checkpoint, checkpoint_path )

    plt.figure()
    plt.title('PSNR')
    plt.plot(range(len(np.array(checkpoint_performance)[:,0])),np.array(checkpoint_performance)[:,0])
    plt.grid(True)
    plt.savefig(os.path.join(opt.outputs_dir, "result.png"))
    plt.close()

    if checkpoint_performance[-1][0]> opt.best:
        print("\t", "This Epoch {} PSNR: {} (BEST)".format(epoch,checkpoint_performance[-1][0]))
        opt.best = checkpoint_performance[-1][0]
        shutil.copyfile(checkpoint_path, os.path.join(opt.outputs_dir,'model_best{}.pth'.format("_amp" if opt.amp else "")))
    else:
        print("\t","This Epoch {} PSNR: {},".format(epoch,checkpoint_performance[-1][0]), 
            "The Best PSNR is {} at Epoch {}".format(opt.best, np.where(np.array(checkpoint_performance)[:,0] == opt.best)))

    # code for recode
    # if opt.amp:
    #     # saving all model
    #     # checkpoint_path = os.path.join(
    #     #     opt.outputs_dir,
    #     #     '{}_epoch_{}_amp.pth'.format(opt.arch, epoch)
    #     #     )
    #     checkpoint_path = os.path.join(opt.outputs_dir,'{}_epoch_last_amp.pth'.format(opt.arch))
    #     checkpoint = {
    #         'model':model.state_dict(),
    #         'optimizer':optimizer.state_dict(),
    #         'epoch':epoch,
    #         'amp':amp.state_dict(),
    #         'best':np.array(checkpoint_performance)[:,0].max(),
    #         'ck_performance':checkpoint_performance,
    #         'ck_time':checkpoint_time
    #     }
    #     torch.save(checkpoint, checkpoint_path )
    #     if checkpoint_performance[-1][0]> opt.best:
    #         print("=> This Epoch {} is the BEST: {}{}".format(epoch,checkpoint_performance[-1][0],opt.process_id))
    #         opt.best = np.array(checkpoint_performance).max()
    #         shutil.copyfile(checkpoint_path, os.path.join(opt.outputs_dir,'model_best_amp.pth'))
    # else:
    #     checkpoint_path = os.path.join(
    #         opt.outputs_dir,
    #         '{}_epoch_{}.pth'.format(opt.arch, epoch)
    #         )
    #     checkpoint = {
    #         'model':model.state_dict(),
    #         'optimizer':optimizer.state_dict(),
    #         'epoch':epoch,
    #         'best':np.array(checkpoint_performance)[:,0].max(),
    #         'ck_performance':checkpoint_performance,
    #         'ck_time':checkpoint_time
    #     }
    #     torch.save(checkpoint, checkpoint_path )
    #     if checkpoint_performance[-1][0] > opt.best:
    #         print("=> This Epoch {} is the BEST: {}".format(epoch,checkpoint_performance[-1][0]))
    #         opt.best = np.array(checkpoint_performance).max()
    #         shutil.copyfile(checkpoint_path, os.path.join(opt.outputs_dir,'model_best.pth'))
    # print("Best:",opt.best,"This Time:",checkpoint_performance[-1][0])

def test_eval(model,dataloader_test,opt):
    psnr_list = []
    ssim_list = []
    filename_list = []

    for batch, (hr,  lr, bicubic, filename) in enumerate(dataloader_test):
        with torch.no_grad():
            if opt.device == "npu":
                lr = lr.npu()
            elif opt.device == "gpu":
                lr = lr.cuda()
            pred = model(lr)

        hr = hr[0].numpy().astype('uint8')
        bicubic = bicubic[0].numpy().astype('uint8')
        pred = np.transpose( (pred[0]*255.0).clamp_(0.0, 255.0).byte().cpu().numpy(),axes = (1,2,0))

        if not opt.multiprocessing_distributed or (opt.multiprocessing_distributed and opt.process_id % opt.ndevices_per_node == 0):
            image_pred = pil_image.fromarray(pred, mode='RGB')
            image_pred.save(os.path.join(opt.outputs_dir, '{}_x{}_output{}.png'.format(filename[0], opt.scale,opt.process_id)))

        image_src = hr/255.0
        image_src = 65.481 * image_src[:,:,0] + 128.553 * image_src[:,:,1] + 24.966 * image_src[:,:,2] + 16 
        image_src = image_src/255.0
        image_src = np.expand_dims(image_src,axis=2)

        bicubic = bicubic/255.0
        bicubic = 65.481 * bicubic[:,:,0] + 128.553 * bicubic[:,:,1] + 24.966 * bicubic[:,:,2] + 16 
        bicubic = bicubic/255.0
        bicubic = np.expand_dims(bicubic,axis=2)

        image_rcan = pred/255.0
        image_rcan = 65.481 * image_rcan[:,:,0] + 128.553 * image_rcan[:,:,1] + 24.966 * image_rcan[:,:,2] + 16 
        image_rcan = image_rcan/255.0
        image_rcan = np.expand_dims(image_rcan,axis=2)

        image_src = image_src[opt.scale+6:-(opt.scale+6),opt.scale+6:-(opt.scale+6),:]
        bicubic = bicubic[opt.scale+6:-(opt.scale+6),opt.scale+6:-(opt.scale+6),:]
        image_rcan = image_rcan[opt.scale+6:-(opt.scale+6),opt.scale+6:-(opt.scale+6),:]

        filename_list.append(filename)
        psnr_list.append(peak_signal_noise_ratio(image_src, image_rcan))
        # psnr_list2.append(peak_signal_noise_ratio(image_src, bicubic))
        ssim_list.append(structural_similarity(image_src, image_rcan,win_size=11,gaussian_weights=True,multichannel=True,data_range=1.0,K1=0.01,K2=0.03,sigma=1.5))
    
    if np.mean(psnr_list)>opt.best:

        if not opt.multiprocessing_distributed or (opt.multiprocessing_distributed and opt.process_id % opt.ndevices_per_node == 0):
            print("\t", "Saving the pictures")
            for filename in filename_list:
                image_path = os.path.join(opt.outputs_dir, '{}_x{}_output{}.png'.format(filename[0], opt.scale,opt.process_id))
                new_image_path = os.path.splitext(image_path)[0]+ "_best"  +os.path.splitext(image_path)[1]
                shutil.copyfile(image_path, new_image_path)
    return np.mean(psnr_list),np.mean(ssim_list)

def train_prepare(opt):
    opt.process_device_map = device_id_to_process_device_map(opt.device_list)
    information_print(opt.process_device_map,mode = 0)

    if opt.multiprocessing_distributed:
        # Since we have ndevices_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        if opt.device_id != None:
            raise ValueError("when you choose multi processing, you don't need to select one npu or gpu")
        opt.ndevices_per_node = len(opt.process_device_map)
        opt.world_size = opt.ndevices_per_node * opt.world_size
        information_print("...multi processing...\nChoose to use {} {}s from device list...".format(opt.ndevices_per_node,opt.device),mode = 0)

    elif opt.device_id != None:
        opt.ndevices_per_node = 1
        information_print("...single processing...\nChoose to use {}ID:{} from device list...".format(opt.device,opt.device_id),mode = 0)
    else:
        raise ValueError("The process_para set Wrong")

    # opt.ndevices_per_node = torch.cuda.device_count()
    # opt.world_size = opt.ndevices_per_node * opt.world_size
    # information_print("Choose to use {} GPU from device count...".format(opt.ndevices_per_node))


    opt.outputs_dir = opt.outputs_dir + "/X" + str(opt.scale)+"/"
    if opt.ifcontinue:
        if not os.path.exists(opt.outputs_dir):
            raise ValueError("Do not find this dir for continuing train, please check")
        if not os.path.exists(os.path.join(opt.outputs_dir,opt.checkpoint_path)):
            raise ValueError("Do not find this file for continuing train, please check")
        print("Continue in {}, train from {}".format(opt.outputs_dir,opt.checkpoint_path))
        opt.checkpoint_path = os.path.join(opt.outputs_dir,opt.checkpoint_path)
    else:
        if not os.path.exists(opt.outputs_dir):
            os.makedirs(opt.outputs_dir)
        else:
            information_print("The dir is existing, if continue, Retraining from and Replacing...",mode = 0)
    # To record the parameters 
    with open(opt.outputs_dir+'/Para_{}.txt'.format(time.strftime("%Y_%m_%d_%H_%M", time.localtime()) ),"w") as f:    
        for i in vars(opt):
            f.write(i+":"+str(vars(opt)[i])+'\n')
    f.close()

def main():
    print("===============main() start=================")
    opt = parser.parse_args()
    information_print(opt,mode = 0)
    if opt.device == "npu":
        import torch
        import torch.npu
    elif opt.device == "gpu":
        import torch
    torch.manual_seed(opt.seed)
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = '29688'
    train_prepare(opt)
    print("===============main() end=================")

    if opt.multiprocessing_distributed:
        # Use torch.multiprocessing.spawn to launch distributed processes: 
        # the main_worker process function
        # the process_id control the selection of individual device (in the device list) 
        # mp.spawn(sub_process_fun,nprocs=numper_of_process,args = args for fun)
        mp.spawn(main_worker, nprocs=opt.ndevices_per_node, args = [opt])
    else:
        # Simply call main_worker function
        main_worker( 0,  opt)

def main_worker(process_id,opt):
    if opt.multiprocessing_distributed:
        # if using mode[multiprocessing_distributed]
        # each sub_process choose device from process_device_map through process_id
        # process_id belongs to [0,...len(process_device_map)-1]
        os.environ['MASTER_ADDR'] = "127.0.0.1"
        os.environ['MASTER_PORT'] = '29688'
        opt.sub_device_id = opt.process_device_map[process_id]
        if opt.device =='npu':
            torch.distributed.init_process_group(backend='hccl', world_size=opt.world_size, rank=process_id) 
        if opt.device == 'gpu':
            torch.distributed.init_process_group(backend='nccl', init_method="env://", world_size=opt.world_size, rank=process_id)
    else:
        opt.sub_device_id = opt.device_id

    opt.process_id = process_id
    if opt.device =='npu':
        loc = 'npu:{}'.format(opt.sub_device_id)
        device = torch.device("npu:{}".format(opt.sub_device_id))
        torch.npu.set_device(device)
        
    if opt.device == 'gpu':
        loc = 'cuda:{}'.format(opt.sub_device_id)
        device = torch.device("cuda:{}".format(opt.sub_device_id))
        torch.cuda.set_device(opt.sub_device_id)

    print("===="*5)
    print("SUB PROCESSING INFORMATION")
    print("Number of Mutil-process {}".format(opt.ndevices_per_node))
    print("rank ID {}".format(process_id))
    print("Wanted device {}ID:{}".format(opt.device,opt.sub_device_id))
    print("Chosen device {}".format(opt.device,torch.cuda.current_device() if opt.device == "gpu" else torch.npu.current_device()))
    print("===="*5)

    criterion = nn.L1Loss()

    if opt.device == "npu":
        model = RCAN(opt).to(device)
    elif opt.device == "gpu":
        model = RCAN(opt).cuda() 

    if opt.device == "npu":    
        optimizer = apex.optimizers.NpuFusedAdam(model.parameters(), lr=opt.lr)
        if opt.amp:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2', loss_scale=32.0, combine_grad=True)
        if opt.multiprocessing_distributed:
            print("start ddp")
            t0 = time.time()
            model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[device], broadcast_buffers=False)
            print("end ddp")
            print("time",time.time()-t0)
    elif opt.device == "gpu":
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
        if opt.amp:
            model, optimizer = amp.initialize(model, optimizer, opt_level = opt.opt_level,loss_scale=opt.loss_scale)
        if opt.multiprocessing_distributed:
            model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[device], broadcast_buffers=False)


    if opt.iffinetuning:
        if os.path.isfile(opt.finetuning_checkpoint_path):
            print("loading checkpoint: {} and finetuning from it".format(opt.finetuning_checkpoint_path)) 
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(opt.finetuning_checkpoint_path, map_location=loc)
            opt.start_epoch = 0
            opt.best = -1
            model.load_state_dict(checkpoint['model'])
            checkpoint_performance = []
            checkpoint_time = []
            print("start finetuning")
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(opt.checkpoint_path))
    elif opt.ifcontinue:
        if os.path.isfile(opt.checkpoint_path):
            print("loading checkpoint: {}".format(opt.checkpoint_path)) 
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(opt.checkpoint_path, map_location=loc)
            opt.start_epoch = len(checkpoint['ck_performance'])
            opt.best = checkpoint['best']
            model.load_state_dict(checkpoint['model'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            checkpoint_performance =checkpoint['ck_performance']
            checkpoint_time =checkpoint['ck_time']
            if opt.amp:
                amp.load_state_dict(checkpoint['amp'])
            print("loaded checkpoint: {} (epoch {})".format(opt.checkpoint_path, checkpoint['epoch']))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(opt.checkpoint_path))
    else:
        opt.start_epoch = 0
        opt.best = -1
        checkpoint_performance = []
        checkpoint_time = []

    torch.backends.cudnn.benchmark=True
    # transform_my=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()])
    transform_my=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()])
    dataset_train = Dataset(opt.train_dataset_dir, opt.patch_size, opt.scale,transform = transform_my)
    if opt.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        opt.workers  = int((opt.workers + opt.ndevices_per_node - 1) / opt.ndevices_per_node)
        opt.batch_size = int(opt.batch_size / opt.ndevices_per_node)
    else:
        train_sampler = None
    dataloader = DataLoader(dataset=dataset_train,batch_size=opt.batch_size,num_workers=opt.workers,pin_memory=True,drop_last=True,sampler=train_sampler)

    # For testing model, every sub-process may carry out one time of testing the whole test dataloader 
    dataset_test = Dataset_test_label(opt.test_dataset_dir, opt.scale)
    # if opt.multiprocessing_distributed:
    #     test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    # else:
    # test_sampler = None
    dataloader_test = DataLoader(dataset=dataset_test,batch_size=1,num_workers=1 ,pin_memory=True,drop_last=True)



    
    for epoch in range(opt.start_epoch, opt.start_epoch + opt.num_epochs):
        # torch.cuda.synchronize()
        if opt.multiprocessing_distributed:
            # To shufful between all epoch 
            train_sampler.set_epoch(epoch)
        model.train()

        if not opt.multiprocessing_distributed or (opt.multiprocessing_distributed and opt.process_id % opt.ndevices_per_node == 0):
            epoch_losses = AverageMeter()
            epoch_timer = AverageMeter()
            timer_iter = timer()
            timer_epoch = timer()
            # with tqdm(total=(len(dataset_train) - len(dataset_train) % (opt.batch_size * opt.ndevices_per_node)) , ncols=100) as _tqdm:
            #     _tqdm.set_description('E:{}/{}(RankID:{})'.format(epoch , opt.start_epoch + opt.num_epochs,opt.process_id))
            #     model.train()
            #     for index,data in enumerate( dataloader):
            #         if index > 3:# the first three inerations don't be recorded in the time analysis
            #             timer_iter.tic()#start timer
            #         inputs, labels, filename = data
            #         if opt.device == "npu":
            #             inputs,labels = inputs.to(device),labels.to(device)
            #         elif opt.device == "gpu":
            #             inputs,labels = inputs.cuda(),labels.cuda()

            #         preds = model(inputs)
            #         loss = criterion(preds, labels)
            #         epoch_losses.update(loss.item(), len(inputs))

            #         optimizer.zero_grad()
            #         if opt.amp:
            #             with amp.scale_loss(loss, optimizer) as scaled_loss:
            #                 scaled_loss.backward()
            #         else:
            #             loss.backward()
            #         optimizer.step()

            #         _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            #         _tqdm.update(len(inputs)*opt.ndevices_per_node)
            #         if index > 3:# the first three inerations don't be recorded in the time analysis
            #             timer_iter.hold()
            #             epoch_timer.update(timer_iter.acc)
            #     model.eval()
            #     psnr_avg,ssim_avg = test_eval(model,dataloader_test,opt)
            #     checkpoint_time.append(epoch_timer.avg)
            #     checkpoint_performance.append([psnr_avg,ssim_avg])
            #     save_checkpoint(opt,epoch,checkpoint_performance,checkpoint_time,model,optimizer)

            print('E:{}/{}(RankID:{})'.format(epoch , opt.start_epoch + opt.num_epochs,opt.process_id))
            model.train()
            timer_epoch.tic()
            for index,data in enumerate(dataloader):
                inputs, labels, filename = data
                if opt.device == "npu":
                    inputs,labels = inputs.to(device),labels.to(device) 
                elif opt.device == "gpu":
                    inputs,labels = inputs.cuda(),labels.cuda()

                if index > 5:# the first three inerations don't be recorded in the time analysis
                    timer_iter.tic()#start timer

                preds = model(inputs)
                loss = criterion(preds, labels)
                epoch_losses.update(loss.item(), len(inputs))
                optimizer.zero_grad()
                if opt.amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
                if index > 5:# the first three inerations don't be recorded in the time analysis
                    timer_iter.hold()
                    epoch_timer.update(timer_iter.release()/(opt.batch_size * opt.ndevices_per_node))
                    # epoch_timer.update(timer_iter.acc/(opt.batch_size * opt.ndevices_per_node))
                if index % 10 == 0 and index > 5:
                    data_length = len(dataset_train) - len(dataset_train) % (opt.batch_size * opt.ndevices_per_node)
                    iteration_all = data_length//(opt.batch_size * opt.ndevices_per_node)
                    information1 = 'Iteration:{}/{}->{:.2f}%'.format(index ,iteration_all,index/iteration_all*100)
                    information2 = 'Time_avg_per_img:{}->FPS:{}'.format(epoch_timer.avg, 1/epoch_timer.avg)
                    information3 = 'Loss:{:.6f}'.format(epoch_losses.avg)
                    print("\t", information1, information2, information3)
            timer_epoch.hold()
            print("\tThis Epoch's whole time: {}".format(timer_epoch.acc))
            model.eval()
            psnr_avg,ssim_avg = test_eval(model,dataloader_test,opt)
            checkpoint_time.append(epoch_timer.avg)
            checkpoint_performance.append([psnr_avg,ssim_avg])
            save_checkpoint(opt,epoch,checkpoint_performance,checkpoint_time,model,optimizer)
        else:
            for index,data in enumerate( dataloader):
                inputs, labels, filename = data
                if opt.device == "npu":
                    inputs,labels = inputs.to(device),labels.to(device)
                elif opt.device == "gpu":
                    inputs,labels = inputs.cuda(),labels.cuda()

                preds = model(inputs)
                loss = criterion(preds, labels)

                optimizer.zero_grad()
                if opt.amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
if __name__ == '__main__':
    main()
    
    
    