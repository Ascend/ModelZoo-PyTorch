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
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os
import apex
from prefetcher import DataPrefetcher
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
import argparse

import numpy as np
import glob
import time
from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from iou_tmp import iiou as iiou_loss
from model import BASNet

import pytorch_ssim
import pytorch_iou
try:
    from apex import amp
except ImportError:
    amp = None

parser = argparse.ArgumentParser()
# Mixed precision training parameters
parser.add_argument('--apex', action='store_true', help='Use apex for mixed precision training')
parser.add_argument('--apex-opt-level', default='O2', type=str,help='For apex mixed precision training'
                        'O0 for FP32 training, O1 for mixed precision training.'
                        'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet')
parser.add_argument('--loss-scale-value', default=1024., type=float,
                    help='loss scale using in amp, default -1 means dynamic')
## for ascend 910
parser.add_argument('--device_id', default=5, type=int, help='device id')
args = parser.parse_args()

device = torch.device(f'npu:{args.device_id}')
torch.npu.set_device(device)
print("Use NPU: {} for training".format(args.device_id))

# ------- 1. define loss function --------

#bce_loss = nn.BCELoss(size_average=True)
bce_loss = nn.BCEWithLogitsLoss(size_average=True).to(device)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def bce_ssim_loss(pred,target):

	bce_out = bce_loss(pred,target)
	ssim_out = 1 - ssim_loss(pred,target)
	iou_out = iiou_loss(pred,target)
	loss = bce_out + ssim_out + iou_out

	return loss

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v):

	loss0 = bce_ssim_loss(d0,labels_v)
	loss1 = bce_ssim_loss(d1,labels_v)
	loss2 = bce_ssim_loss(d2,labels_v)
	loss3 = bce_ssim_loss(d3,labels_v)
	loss4 = bce_ssim_loss(d4,labels_v)
	loss5 = bce_ssim_loss(d5,labels_v)
	loss6 = bce_ssim_loss(d6,labels_v)
	loss7 = bce_ssim_loss(d7,labels_v)
	#ssim0 = 1 - ssim_loss(d0,labels_v)

	# iou0 = iou_loss(d0,labels_v)
	#loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7#+ 5.0*lossa
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item(),loss6.item()))
	# print("BCE: l1:%3f, l2:%3f, l3:%3f, l4:%3f, l5:%3f, la:%3f, all:%3f\n"%(loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],lossa.data[0],loss.data[0]))

	return loss0, loss


# ------- 2. set the directory of training dataset --------

data_dir = './train_data/'
tra_image_dir = 'DUTS/DUTS-TR/DUTS-TR/im_aug/'
tra_label_dir = 'DUTS/DUTS-TR/DUTS-TR/gt_aug/'

image_ext = '.jpg'
label_ext = '.png'

model_dir = "./saved_models/basnet_bsi/"


epoch_num = 100000
batch_size_train = 8 
batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
for img_path in tra_img_name_list:
	img_name = img_path.split("/")[-1]

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

print("---")
print("train_dir: ", data_dir + tra_image_dir)
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(256),
        RandomCrop(224),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=False, num_workers=128,pin_memory=True,drop_last=True)
#data_prefetcher_stream = torch.npu.Stream()
#prefetch = DataPrefetcher(salobj_dataloader,stream=data_prefetcher_stream)
# ------- 3. define model --------
# define the net
net = BASNet(3, 1)
if torch.npu.is_available():
    #net.npu()
    net.to(device)

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = apex.optimizers.NpuFusedAdam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)  # strongly increased velocity
#optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
if args.apex:
    net, optimizer = amp.initialize(net, optimizer,
                                      opt_level=args.apex_opt_level,                             
                                      combine_grad=True) #loss_scale=args.loss_scale_value,

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0

for epoch in range(0, epoch_num):
    net.train()
    init_time = time.time()
    for i, data in enumerate(salobj_dataloader):
        start_time = time.time()
        data_time = start_time - init_time
        print('data_loader_time:',data_time)
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']
        

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        #inputs = inputs.to(device)
        #labels = labels.to(device)

        # wrap them in Variable
        if torch.npu.is_available():
            inputs_v, labels_v = Variable(inputs.npu(), requires_grad=False), Variable(labels.npu(),
                                                                                        requires_grad=False)
            #inputs_v, labels_v = Variable(inputs.to(device), requires_grad=False), Variable(labels.to(device),
                                                                                        #requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        d0, d1, d2, d3, d4, d5, d6, d7 = net(inputs_v)

        print('format',d7.shape,labels_v.shape)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v)
        print("the loss is:", loss, "the loss2 is:", loss2)
        if args.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        #loss.backward()
        optimizer.step()
        train_time = time.time()-start_time
        print('train_time:',train_time)
        fps = batch_size_train / (time.time() - start_time)

        # # print statistics
        running_loss += loss.item()
        running_tar_loss += loss2.item()

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, d7, loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, fps: %3f" % (
        epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val, fps))

        if ite_num % 200 == 0:  # save model every 200 iterations

            torch.save(net.state_dict(), model_dir + "basnet_bsi_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0
        
        init_time = time.time()

print('-------------Congratulations! Training Done!!!-------------')
