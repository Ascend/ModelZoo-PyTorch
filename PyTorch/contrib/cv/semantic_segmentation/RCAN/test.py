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
if torch.__version__ >= "1.8":
    import torch_npu
from torch import nn
import torch.multiprocessing as mp
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from model import RCAN
from dataset import Dataset,Dataset_test_label
from utils import AverageMeter,device_id_to_process_device_map,information_print,timer
import shutil
import numpy as np 
import time 
import PIL.Image as pil_image
import skimage.io as io
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
from apex import amp


parser = argparse.ArgumentParser()
# train and test setting
parser.add_argument('--arch', type=str, default='RCAN')
parser.add_argument('--test_dataset_dir', type=str, required=True)
parser.add_argument('--outputs_dir', type=str, required=True)
parser.add_argument('--workers', type=int, default=8)

# model setting
parser.add_argument('--scale', type=int, required=True)
parser.add_argument('--num_features', type=int, default=64)
parser.add_argument('--num_rg', type=int, default=10)
parser.add_argument('--num_rcab', type=int, default=20)
parser.add_argument('--reduction', type=int, default=16)

# test
parser.add_argument('--checkpoint_path', type=str, help='the path of checkpoint to load')

# amp setting
parser.add_argument('--amp', default=False, action='store_true', help='if use amp to train the model')
parser.add_argument('--loss_scale', default=128.0, type=float, help='amp setting: loss scale, default 128.0')
parser.add_argument('--opt_level', default='O2', type=str, help='amp setting: opt level, default O2')

# device and process setting
parser.add_argument('--device', type=str,default= "gpu", help='npu or gpu')
parser.add_argument('--device_list', type=str,default= '0,1,2,3,4,5,6,7')
parser.add_argument('--device_id', type=int, default=None,help='index of gpu/npu to use')
parser.add_argument('--world_size', type=int, default=1,help='number of nodes for distributed training')
parser.add_argument('--from_multiprocessing_distributed', action='store_true',
                    help='if the checkpoint trained from mutil P')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

def test_eval(model,dataloader_test,opt):
    psnr_list = []
    psnr_list_bic = []
    ssim_list = []
    ssim_list_bic = []

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

        psnr_list.append(peak_signal_noise_ratio(image_src, image_rcan))
        psnr_list_bic.append(peak_signal_noise_ratio(image_src, bicubic))
        ssim_list.append(structural_similarity(image_src, image_rcan,win_size=11,gaussian_weights=True,multichannel=True,data_range=1.0,K1=0.01,K2=0.03,sigma=1.5))
        ssim_list_bic.append(structural_similarity(image_src, bicubic,win_size=11,gaussian_weights=True,multichannel=True,data_range=1.0,K1=0.01,K2=0.03,sigma=1.5))
    return np.mean(psnr_list),np.mean(ssim_list),np.mean(psnr_list_bic),np.mean(ssim_list_bic)

def test_prepare(opt):
    opt.process_device_map = device_id_to_process_device_map(opt.device_list)
    information_print(opt.process_device_map,mode = 0)

    if opt.multiprocessing_distributed:
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

    opt.outputs_dir = opt.outputs_dir + "/X{}/".format(
        str(opt.scale)) 
    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)
    else:
        information_print("The dir is existing, if continue, Retraining from and Replacing...",mode = 0)
    # To record the parameters 
    with open(opt.outputs_dir+'/TEST_Para_{}.txt'.format(time.strftime("%Y_%m_%d_%H_%M", time.localtime()) ),"w") as f:    
        for i in vars(opt):
            f.write(i+":"+str(vars(opt)[i])+'\n')
    f.close()

def main():
    print("===============main() start=================")
    opt = parser.parse_args()
    information_print(opt,mode = 0)
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = '29688'
    test_prepare(opt)
    print("===============main() end=================")

    if opt.device == "npu":
        import torch
        import torch.npu
    elif opt.device == "gpu":
        import torch

    if opt.multiprocessing_distributed:
        # Use torch.multiprocessing.spawn to launch distributed processes: 
        # the main_worker process function
        # the process_id control the selection of individual device (in the device list) 
        # mp.spawn(sub_process_fun,nprocs=numper_of_process,args = args for fun)
        mp.spawn(main_worker, nprocs=opt.ndevices_per_node,args=[ opt])
    else:
        # Simply call main_worker function
        main_worker( 0, opt)

def main_worker(process_id,opt):
    opt.process_id = process_id
    if opt.multiprocessing_distributed:
        os.environ['MASTER_ADDR'] = "127.0.0.1"
        os.environ['MASTER_PORT'] = '29688'
        opt.device_id = opt.process_device_map[opt.process_id]
        if opt.device =='npu':
            torch.distributed.init_process_group(backend='hccl', world_size=opt.world_size, rank=opt.process_id) 
        if opt.device == 'gpu':
            torch.distributed.init_process_group(backend='nccl', init_method="env://", world_size=opt.world_size, rank=opt.process_id)
    else:
        pass

    print("===="*5)
    print("SUB PROCESSING INFORMATION")
    print("Number of Mutil-process {}".format(opt.ndevices_per_node))
    print("rank ID {}".format(opt.process_id))
    print("Wanted device {}ID:{}".format(opt.device,opt.device_id))
    print("Chosen device {}".format(opt.device,torch.cuda.current_device() if opt.device == "gpu" else torch.npu.current_device()))
    print("===="*5)

    if opt.device =='npu':
        loc = 'npu:{}'.format(opt.device_id)
        device = torch.device("npu:{}".format(opt.device_id))
        torch.npu.set_device(device)
        model = RCAN(opt).to(device)
    if opt.device == 'gpu':
        loc = 'cuda:{}'.format(opt.device_id)
        torch.cuda.set_device(opt.device_id)
        model = RCAN(opt).cuda() 

    if opt.amp:
        model = amp.initialize(model, opt_level = opt.opt_level,loss_scale=opt.loss_scale)
    if opt.multiprocessing_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[opt.device_id], broadcast_buffers=False)

    if os.path.exists(opt.checkpoint_path):
        print("loading checkpoint {}".format(opt.checkpoint_path)) 
        # Map model to be loaded to specified single gpu.
        checkpoint = torch.load(opt.checkpoint_path, map_location=loc)
        # print(checkpoint.keys())
        if not opt.from_multiprocessing_distributed:
            # load from checkpoint trained at 1P 
            # load_state_dict from checkpoint['model'] 
            if opt.multiprocessing_distributed:
                pretrained_dict = checkpoint['model']
                new_stat_dict = {}
                for k,v in pretrained_dict.items():
                    new_stat_dict["module."+k] = v
                model.load_state_dict(new_stat_dict)
            else:
                model.load_state_dict(checkpoint['model'])
        else:
            if opt.multiprocessing_distributed:
                model.load_state_dict(checkpoint['model'])
            else:
                # load from checkpoint trained at 8P 
                # load_state_dict from new_stat_dict built from checkpoint['model']
                # key in new_stat_dict = key[7:] in checkpoint['model'] (remove the start str 'module.')
                pretrained_dict = checkpoint['model']
                new_stat_dict = {}
                for k,v in pretrained_dict.items():
                    new_stat_dict[k[7:]] = v
                model.load_state_dict(new_stat_dict)

        checkpoint_performance =checkpoint['ck_performance']
        checkpoint_time =checkpoint['ck_time']
        if opt.amp:
            amp.load_state_dict(checkpoint['amp'])
        print("loaded checkpoint '{}' (epoch {})".format(opt.checkpoint_path, len(checkpoint['ck_performance'])))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(opt.checkpoint_path))


    torch.backends.cudnn.benchmark=True

    # For testing model, every sub-process may carry out one time of testing the whole test dataloader 
    dataset_test = Dataset_test_label(opt.test_dataset_dir, opt.scale)
    if opt.multiprocessing_distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        test_sampler = None
    dataloader_test = DataLoader(dataset=dataset_test,batch_size=1,num_workers=1 ,pin_memory=True,drop_last=True)

    model.eval()
    psnr_avg, ssim_avg, psnr_avg_bic, ssim_avg_bic = test_eval(model,dataloader_test,opt)
    print("PSNR_bic:",psnr_avg_bic)
    print("SSIM_bic:",ssim_avg_bic)
    print("PSNR_RCAN:",psnr_avg)
    print("SSIM_RCAN:",ssim_avg)
    print("TIME:",np.mean(checkpoint_time))

if __name__ == '__main__':
    main()
