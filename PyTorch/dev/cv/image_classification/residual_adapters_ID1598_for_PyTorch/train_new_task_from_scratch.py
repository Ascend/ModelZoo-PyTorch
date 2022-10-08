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
# train_new_task_from_scratch.py
# created by Sylvestre-Alvise Rebuffi [srebuffi@robots.ox.ac.uk]
# Copyright 漏 The University of Oxford, 2017-2020
# This code is made available under the Apache v2.0 licence, see LICENSE.txt for details

import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import models
import os
import time
import argparse
import numpy as np

from torch.autograd import Variable

import imdbfolder_coco as imdbfolder
import config_task
import utils_pytorch
import sgd
try:
    from apex import amp
except ImportError:
    apex = None

parser = argparse.ArgumentParser(description='PyTorch Residual Adapters training')
parser.add_argument('--dataset', default='cifar100', nargs='+', help='Task(s) to be trained')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--wd', default=1., type=float, help='weight decay for the classification layer')
parser.add_argument('--wd3x3', default=1., type=float, nargs='+', help='weight decay for the 3x3')
parser.add_argument('--wd1x1', default=1., type=float, nargs='+', help='weight decay for the 1x1')
parser.add_argument('--nb_epochs', default=120, type=int, help='nb epochs')
parser.add_argument('--step1', default=80, type=int, help='nb epochs before first lr decrease')
parser.add_argument('--step2', default=100, type=int, help='nb epochs before second lr decrease')
parser.add_argument('--mode', default='parallel_adapters', type=str, help='Task adaptation mode')
parser.add_argument('--proj', default='11', type=str, help='Position of the adaptation module')
parser.add_argument('--dropout', default='00', type=str, help='Position of dropouts')
parser.add_argument('--expdir', default='/scratch/shared/nfs1/srebuffi/exp/dem_learning/tmp/', help='Save folder')
parser.add_argument('--datadir', default='/scratch/local/ramdisk/srebuffi/decathlon/', help='folder containing data folder')
parser.add_argument('--imdbdir', default='/scratch/local/ramdisk/srebuffi/decathlon/annotations/', help='annotation folder')
parser.add_argument('--source', default='/scratch/shared/nfs1/srebuffi/exp/dem_learning/C100_alone/checkpoint/ckptpost11bnresidual11cifar1000.000180607060.t7', type=str, help='Network source')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--factor', default='1.', type=float, help='Width factor of the network')
# Mixed precision training parameters
parser.add_argument('--apex', action='store_true', help='Use apex for mixed precision training')
parser.add_argument('--apex-opt-level', default='O1', type=str,help='For apex mixed precision training'
                        'O0 for FP32 training, O1 for mixed precision training.'
                        'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet')
parser.add_argument('--loss-scale-value', default=1024., type=float,
                    help='loss scale using in amp, default -1 means dynamic')
# for ascend 910
parser.add_argument('--device_id', default=5, type=int, help='device id')
args = parser.parse_args()
args.archi ='default'
config_task.mode = args.mode
config_task.proj = args.proj
config_task.factor = args.factor
args.use_npu = torch.npu.is_available()

device = torch.device(f'npu:{args.device_id}')
torch.npu.set_device(device)
print("Use NPU: {} for training".format(args.device_id))

if type(args.dataset) is str:
    args.dataset = [args.dataset]

if type(args.wd3x3) is float:
    args.wd3x3 = [args.wd3x3]

if type(args.wd1x1) is float:
    args.wd1x1 = [args.wd1x1]

if not os.path.isdir(args.expdir):
    os.mkdir(args.expdir) 

config_task.decay3x3 = np.array(args.wd3x3) * 0.0001
config_task.decay1x1 = np.array(args.wd1x1) * 0.0001
args.wd = args.wd *  0.0001

args.ckpdir = args.expdir + '/checkpoint/'
args.svdir  = args.expdir + '/results/'

if not os.path.isdir(args.ckpdir):
    os.mkdir(args.ckpdir) 

if not os.path.isdir(args.svdir):
    os.mkdir(args.svdir) 

config_task.isdropout1 = (args.dropout[0] == '1')
config_task.isdropout2 = (args.dropout[1] == '1')

#####################################

# Prepare data loaders
train_loaders, val_loaders, num_classes = imdbfolder.prepare_data_loaders(args.dataset,args.datadir,args.imdbdir,True)
args.num_classes = num_classes

# Create the network
net = models.resnet26(num_classes)


start_epoch = 0
best_acc = 0  # best test accuracy
results = np.zeros((4,start_epoch+args.nb_epochs,len(args.num_classes)))
all_tasks = range(len(args.dataset))
np.random.seed(1993)

if args.use_npu:
    net.npu()
    cudnn.benchmark = True


args.criterion = nn.CrossEntropyLoss()
optimizer = sgd.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.wd)

if args.apex:
    net, optimizer = amp.initialize(net, optimizer,
                                      opt_level=args.apex_opt_level,
                                      loss_scale=args.loss_scale_value,
                                      combine_grad=True)

print("Start training")
for epoch in range(start_epoch, start_epoch+args.nb_epochs):
    training_tasks = utils_pytorch.adjust_learning_rate_and_learning_taks(optimizer, epoch, args)
    st_time = time.time()
    
    # Training and validation
    train_acc, train_loss = utils_pytorch.train(epoch, train_loaders, training_tasks, net, args, optimizer)
    test_acc, test_loss, best_acc = utils_pytorch.test(epoch,val_loaders, all_tasks, net, best_acc, args, optimizer)
        
    # Record statistics
    for i in range(len(training_tasks)):
        current_task = training_tasks[i]
        results[0:2,epoch,current_task] = [train_loss[i].cpu(),train_acc[i].cpu()]
    for i in all_tasks:
        results[2:4,epoch,i] = [test_loss[i].cpu(),test_acc[i].cpu()]
    np.save(args.svdir+'/results_'+'adapt'+str(args.seed)+args.dropout+args.mode+args.proj+''.join(args.dataset)+'wd3x3_'+str(args.wd3x3)+'_wd1x1_'+str(args.wd1x1)+str(args.wd)+str(args.nb_epochs)+str(args.step1)+str(args.step2),results)
    print('Epoch lasted {0}'.format(time.time()-st_time))

