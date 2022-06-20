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
from __future__ import print_function

import os
import argparse
import socket
import time
import sys
from tqdm import tqdm
import mkl

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models import model_pool
from models.util import create_model, get_teacher_name

from distill.util import Embed
from distill.criterion import DistillKL, NCELoss, Attention, HintLoss

from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from dataset.cifar import CIFAR100, MetaCIFAR100
from dataset.transform_cfg import transforms_options, transforms_list

from util import adjust_learning_rate, accuracy, AverageMeter, rotrate_concat, Logger, generate_final_report
from eval.meta_eval import meta_test, meta_test_tune
from eval.cls_eval import validate
from losses import simple_contrstive_loss

import numpy as np
import wandb
from dataloader import get_dataloaders
import copy
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

os.environ["CUDA_VISIBLE_DEVICES"]
mkl.set_num_threads(2)


class Wrapper(nn.Module):

    def __init__(self, model, args):
        super(Wrapper, self).__init__()
    
        self.model = model
        self.feat = torch.nn.Sequential(*list(self.model.children())[:-2])
        
        self.last = torch.nn.Linear(list(self.model.children())[-2].in_features, 64)       
        
    def forward(self, images):
        feat = self.feat(images)
        feat = feat.view(images.size(0), -1)
        out = self.last(feat)
        
        return feat, out
    
    
    
def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--eval_freq', type=int, default=10, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset and model
    parser.add_argument('--model_s', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--model_t', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100'])
    parser.add_argument('--simclr', type=bool, default=False, help='use simple contrastive learning representation')
    parser.add_argument('--ssl', type=bool, default=True, help='use self supervised learning')
    parser.add_argument('--tags', type=str, default="gen1, ssl", help='add tags for the experiment')
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--use_trainval', type=bool, help='use trainval set')

    # path to teacher model
    parser.add_argument('--path_t', type=str, default="", help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'contrast', 'hint', 'attention'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    # parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # cosine annealing
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')

    # specify folder
    parser.add_argument('--model_path', type=str, default='save/', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='tb/', help='path to tensorboard')
    parser.add_argument('--data_root', type=str, default='/raid/data/IncrementLearn/imagenet/Datasets/MiniImagenet/', help='path to data root')

    # setting for meta-learning
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')

    #memory hyper parameters
    parser.add_argument('--gamma', type=float, default=1.0, help='loss cofficient for ssl loss')
    parser.add_argument('--contrast_temp', type=float, default=1.0, help='temperature for contrastive ssl loss')
    parser.add_argument('--membank_size', type=int, default=6400, help='temperature for contrastive ssl loss')
    parser.add_argument('--memfeature_size', type=int, default=64, help='temperature for contrastive ssl loss')
    parser.add_argument('--mvavg_rate', type=float, default=0.99, help='temperature for contrastive ssl loss')
    parser.add_argument('--trans', type=int, default=16, help='number of transformations')
    parser.add_argument('--w_ce', type=float, default=1.0, help='loss cofficient for ce loss')
    parser.add_argument('--w_div', type=float, default=1.0, help='loss cofficient for divergence loss')
    parser.add_argument('--pretrained_path', type=str, default="", help='student pretrained path')

    opt = parser.parse_args()

    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'

    if opt.use_trainval:
        opt.trial = opt.trial + '_trainval'

    # set the path according to the environment
    if not opt.model_path:
        opt.model_path = './models_distilled'
    if not opt.tb_path:
        opt.tb_path = './tensorboard'
    if not opt.data_root:
        opt.data_root = './data/{}'.format(opt.dataset)
    else:
        opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
    opt.data_aug = True
    
    tags = opt.tags.split(',')
    opt.tags = list([])
    for it in tags:
        opt.tags.append(it)
        
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_trans_{}_tag_{}'.format(opt.model_s, opt.model_t, opt.dataset,
                                                                      opt.distill, opt.gamma, opt.alpha, opt.beta,
                                                                      opt.transform, opt.tags[-1])

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    opt.model_name = '{}_{}'.format(opt.model_name, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    
    #extras
    opt.fresh_start = True

    return opt


def load_teacher(model_path, model_name, n_cls, dataset='miniImageNet', trans=16, embd_size=64):
    """load the teacher model"""
    print('==> loading teacher model')
    print(model_name)
    model = create_model(model_name, n_cls, dataset, n_trans=trans, embd_sz=embd_size)
    if torch.npu.device_count() > 1:
        print("gpu count:", torch.npu.device_count())
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model


def main():
    best_acc = 0

    opt = parse_option()
    wandb.init(project=opt.model_path.split("/")[-1], tags=opt.tags)
    wandb.config.update(opt)
    wandb.save('*.py')
    wandb.run.save()
        
        
    # dataloader
    train_loader, val_loader, meta_testloader, meta_valloader, n_cls, no_sample = get_dataloaders(opt)
    
    # model
    model_t = []
    if("," in opt.path_t):
        for path in opt.path_t.split(","):
            model_t.append(load_teacher(path, opt.model_t, n_cls, opt.dataset, opt.trans, opt.memfeature_size))
    else:
        model_t.append(load_teacher(opt.path_t, opt.model_t, n_cls, opt.dataset, opt.trans, opt.memfeature_size))
    
    model_s = create_model(opt.model_s, n_cls, opt.dataset, n_trans=opt.trans, embd_sz=opt.memfeature_size)
    if torch.npu.device_count() > 1:
        print("second gpu count:", torch.npu.device_count())
        model_s = nn.DataParallel(model_s)
    if opt.pretrained_path != "":
        model_s.load_state_dict(torch.load(opt.pretrained_path)['model'])
    wandb.watch(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    criterion_kd = DistillKL(opt.kd_T)

    optimizer = optim.SGD(model_s.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    if torch.npu.is_available():
        for m in model_t: 
            m.npu()
        model_s.npu()
        criterion_cls = criterion_cls.npu()
        criterion_div = criterion_div.npu()
        criterion_kd = criterion_kd.npu()
        cudnn.benchmark = True
    
    MemBank = np.random.randn(no_sample, opt.memfeature_size)
    MemBank = torch.tensor(MemBank, dtype=torch.float).npu()
    MemBankNorm = torch.norm(MemBank, dim=1, keepdim=True)
    MemBank = MemBank / (MemBankNorm + 1e-6)
           
    meta_test_acc = 0 
    meta_test_std = 0
    # routine: supervised model distillation
    for epoch in range(1, opt.epochs + 1):

        if opt.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss, MemBank = train(epoch, train_loader, model_s, model_t , criterion_cls, criterion_div, criterion_kd, optimizer, opt, MemBank)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        val_acc = 0
        val_loss = 0
        meta_val_acc = 0
        meta_val_std = 0
#         val_acc, val_acc_top5, val_loss = validate(val_loader, model_s, criterion_cls, opt)
        
        
#         #evaluate
#         start = time.time()
#         meta_val_acc, meta_val_std = meta_test(model_s, meta_valloader)
#         test_time = time.time() - start
#         print('Meta Val Acc: {:.4f}, Meta Val std: {:.4f}, Time: {:.1f}'.format(meta_val_acc, meta_val_std, test_time))
        
        #evaluate
        start = time.time()
        meta_test_acc, meta_test_std = 0,0 #meta_test(model_s, meta_testloader, use_logit=False)
        test_time = time.time() - start
        print('Meta Test Acc: {:.4f}, Meta Test std: {:.4f}, Time: {:.1f}'.format(meta_test_acc, meta_test_std, test_time))
        
        
        # regular saving
        if epoch % opt.save_freq == 0 or epoch==opt.epochs:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
            }            
            save_file = os.path.join(opt.save_folder, 'model_'+str(wandb.run.name)+'.pth')
            torch.save(state, save_file)
            
            #wandb saving
            torch.save(state, os.path.join(wandb.run.dir, "model.pth"))
        
        wandb.log({'epoch': epoch, 
                   'Train Acc': train_acc,
                   'Train Loss':train_loss,
                   'Val Acc': val_acc,
                   'Val Loss':val_loss,
                   'Meta Test Acc': meta_test_acc,
                   'Meta Test std': meta_test_std,
                   'Meta Val Acc': meta_val_acc,
                   'Meta Val std': meta_val_std
                  })        
        
    #final report
    print("GENERATING FINAL REPORT")
    generate_final_report(model_s, opt, wandb)
    
    #remove output.txt log file 
    output_log_file = os.path.join(wandb.run.dir, "output.log")
    if os.path.isfile(output_log_file):
        os.remove(output_log_file)
    else:    ## Show an error ##
        print("Error: %s file not found" % output_log_file)
        

def train(epoch, train_loader, model_s, model_t , criterion_cls, criterion_div, criterion_kd, optimizer, opt, MemBank):
    """One epoch training"""
    model_s.train()
    for m in model_t:
        m.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    train_indices = list(range(len(MemBank)))

    end = time.time()
    
    with tqdm(train_loader, total=len(train_loader)) as pbar:
        for idx, (input, input2, input3, input4, target, indices) in enumerate(pbar):
            data_time.update(time.time() - end)

            input = input.float()
            if torch.npu.is_available():
                input = input.npu()
                input2 = input2.npu()
                input3 = input3.npu()
                input4 = input4.npu()
                target = target.npu()
                indices = indices.npu()
            batch_size = input.shape[0]

            generated_data = rotrate_concat([input, input2, input3, input4])
            train_targets = target.repeat(opt.trans)
            proxy_labels = torch.zeros(opt.trans*batch_size).npu().long()

            for ii in range(opt.trans):
                proxy_labels[ii*batch_size:(ii+1)*batch_size] = ii

            # ===================forward=====================
            with torch.no_grad():
                (_,_,_,_, feat_t), (train_logit_t, eq_logit_t, inv_rep_t) = model_t[0](generated_data, inductive=True)
            (_,_,_,_, feat_s), (train_logit_s, eq_logit_s, inv_rep_s) = model_s(generated_data, inductive=True)

            # ===================memory bank of negatives for current batch=====================
            np.random.shuffle(train_indices)
            mn_indices_all = np.array(list(set(train_indices) - set(indices)))
            np.random.shuffle(mn_indices_all)
            mn_indices = mn_indices_all[:opt.membank_size]
            mn_arr = MemBank[mn_indices]
            mem_rep_of_batch_imgs = MemBank[indices]

            loss_ce = criterion_cls(train_logit_s, train_targets)
            loss_eq = criterion_cls(eq_logit_s, proxy_labels)
            loss_div = criterion_div(train_logit_s, train_logit_t)
            loss_div_eq = criterion_div(eq_logit_s, eq_logit_t)
            loss_mse_inv = torch.nn.functional.mse_loss(inv_rep_s, inv_rep_t)
            loss_mse_feat = torch.nn.functional.mse_loss(feat_s, feat_t)

            inv_rep_0 = inv_rep_s[:batch_size, :]
            loss_inv = simple_contrstive_loss(mem_rep_of_batch_imgs, inv_rep_0, mn_arr, opt.contrast_temp)
            for ii in range(1, opt.trans):
                loss_inv += simple_contrstive_loss(inv_rep_0, inv_rep_s[(ii*batch_size):((ii+1)*batch_size), :], mn_arr, opt.contrast_temp)
            loss_inv = loss_inv/opt.trans

            loss = opt.w_ce * (opt.gamma * (loss_eq + loss_inv) + loss_ce) + opt.w_div*(loss_div + loss_div_eq + loss_mse_inv + loss_mse_feat)
            
            acc1, acc5 = accuracy(train_logit_s, train_targets, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # ===================update memory bank======================
            MemBankCopy = MemBank.clone().detach()
            MemBankCopy[indices] = (opt.mvavg_rate * MemBankCopy[indices]) + ((1 - opt.mvavg_rate) * inv_rep_0)
            MemBank = MemBankCopy.clone().detach()
            
            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()
            
            pbar.set_postfix({"Acc@1":'{0:.2f}'.format(top1.avg.cpu().numpy()), 
                              "Acc@5":'{0:.2f}'.format(top5.avg.cpu().numpy(),2), 
                              "Loss" :'{0:.2f}'.format(losses.avg,2), 
                             })
            
            
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg, MemBank
   
    
if __name__ == '__main__':
    main()
