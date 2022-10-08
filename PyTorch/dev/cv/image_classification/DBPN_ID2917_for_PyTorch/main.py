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
import argparse
from math import log10

import os
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dbpn import Net as DBPN
from dbpn_v1 import Net as DBPNLL
from dbpns import Net as DBPNS
from dbpn_iterative import Net as DBPNITER
from data import get_training_set
import torch.distributed as dist
import pdb
import socket
import time
import torch.npu
import os
from data import get_eval_set
import apex
from apex import amp
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=8, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=50, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./Dataset')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--hr_train_dataset', type=str, default='DIV2K_train_HR')
parser.add_argument('--model_type', type=str, default='DBPNLL')
parser.add_argument('--residual', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=40, help='Size of cropped HR image')
parser.add_argument('--pretrained_sr', default='MIX2K_LR_aug_x4dl10DBPNITERtpami_epoch_399.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='tpami_residual_filter8', help='Location to save checkpoint models')
parser.add_argument('--input_dir', type=str, default='Input')
parser.add_argument('--test_dataset', type=str, default='Set5_LR_x8')
parser.add_argument('--device_id', default=0, type=int, help='device id')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='env://',
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='hccl', type=str,
                    help='distributed backend')
parser.add_argument('--dist-rank',
                    default=0,
                    type=int,
                    help='node rank for distributed training')
parser.add_argument('--distributed',
                    action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs.')



opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)


def train(epoch):



    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        start_time = time.time()
        input, target, bicubic = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        if cuda:
            input = input.npu(f'npu:{NPU_CALCULATE_DEVICE}',non_blocking=True)
            target = target.npu(f'npu:{NPU_CALCULATE_DEVICE}',non_blocking=True)
            bicubic = bicubic.npu(f'npu:{NPU_CALCULATE_DEVICE}',non_blocking=True)

        optimizer.zero_grad()
        t0 = time.time()
        prediction = model(input)

        if opt.residual:
            prediction = prediction + bicubic

        loss = criterion(prediction, target)
        t1 = time.time()
        epoch_loss += loss.data
        #loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        step_time = time.time() - start_time
        if opt.distributed:
            FPS = (opt.batchSize* opt.rank_size) / step_time
        else:
            FPS = opt.batchSize / step_time
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec, time/step(s):{:.4f}, FPS:{:.3f}".format(epoch, iteration, len(training_data_loader), loss.data, (t1 - t0),step_time,FPS))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    avg_psnr = 0
    model.eval()
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.npu(f'npu:{NPU_CALCULATE_DEVICE}')
            target = target.npu(f'npu:{NPU_CALCULATE_DEVICE}')

        prediction = model(input)
        mse = criterion(prediction, target)
        #psnr = 10 * log10(1 / mse.data[0])
        psnr = 10 * log10(1 / mse.item())
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def checkpoint(epoch):
    model_out_path = opt.save_folder+opt.hr_train_dataset+hostname+opt.model_type+opt.prefix+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

cuda = opt.gpu_mode
if cuda and not torch.npu.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.npu.manual_seed(opt.seed)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29688'
    if opt.distributed:
        if 'RANK_SIZE' in os.environ and 'RANK_ID' in os.environ:
            opt.rank_size = int(os.environ['RANK_SIZE'])
            opt.rank_id = int(os.environ['RANK_ID'])
            opt.rank = opt.dist_rank * opt.rank_size + opt.rank_id
            opt.world_size = opt.world_size * opt.rank_size
            opt.device_id = opt.rank_id
            dist.init_process_group(backend=opt.dist_backend,
                                    world_size=opt.world_size, rank=opt.rank)
        else:
            raise RuntimeError("Please set RANK_SIZE and RANK_ID for .")
    device = torch.device(f'npu:{opt.device_id}')
    torch.npu.set_device(device)
print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.hr_train_dataset, opt.upscale_factor, opt.patch_size, opt.data_augmentation)
if opt.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
else:
    train_sampler = None
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,sampler=train_sampler,shuffle=(train_sampler is None))
test_set = get_eval_set(os.path.join(opt.input_dir,opt.test_dataset), opt.upscale_factor)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)


print('===> Building model ', opt.model_type)
if opt.model_type == 'DBPNLL':
    model = DBPNLL(num_channels=3, base_filter=64,  feat = 256, num_stages=10, scale_factor=opt.upscale_factor) 
elif opt.model_type == 'DBPN-RES-MR64-3':
    model = DBPNITER(num_channels=3, base_filter=64,  feat = 256, num_stages=3, scale_factor=opt.upscale_factor)
else:
    model = DBPN(num_channels=3, base_filter=64,  feat = 256, num_stages=7, scale_factor=opt.upscale_factor) 
    
#model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion = nn.L1Loss()

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=f'npu:{NPU_CALCULATE_DEVICE}'))
        print('Pre-trained SR model is loaded.')

if cuda:
    model = model.npu(f'npu:{NPU_CALCULATE_DEVICE}')
    criterion = criterion.npu(f'npu:{NPU_CALCULATE_DEVICE}')

optimizer = apex.optimizers.NpuFusedAdam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
model, optimizer = amp.initialize(model, optimizer, opt_level="O2",loss_scale=128.0,combine_grad=True)
if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.device_id])
for epoch in range(opt.start_iter, opt.nEpochs + 1):
    if opt.distributed:
        train_sampler.set_epoch(epoch)
    train(epoch)
    test()

    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch+1) % (opt.nEpochs/2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
            
    if (epoch+1) % (opt.snapshots) == 0:
        checkpoint(epoch)
