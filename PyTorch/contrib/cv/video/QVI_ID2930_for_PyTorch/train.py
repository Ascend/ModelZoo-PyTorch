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
import models
import datas

import argparse
import torch
import torchvision
import torchvision.transforms as TF
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from math import log10
import numpy as np
import datetime
from utils.config import Config
from tensorboardX import SummaryWriter
import sys

print(f"initial network on cpu, it might take minutes.")

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="configs/train_config")
parser.add_argument("--data_url", type=str, default="datasets")
args = parser.parse_args()

config = Config.from_file(args.config)
CALCULATE_DEVICE = "npu: 0"
torch.npu.set_device(CALCULATE_DEVICE)
a = torch.linspace(-1.0, 1.0, 12).npu()
vgg_pth = args.data_url + "/vgg16-397923af.pth"
pwc_pth = args.data_url + "/pwc-checkpoint.pt"
config.trainset_root = args.data_url + config.trainset_root
config.validationset_root = args.data_url + config.validationset_root

vgg16 = torchvision.models.vgg16()
vgg16.load_state_dict(torch.load(vgg_pth, map_location="cpu"))
vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
vgg16_conv_4_3 = nn.DataParallel(vgg16_conv_4_3)

# preparing transform & datasets
normalize1 = TF.Normalize(config.mean, [1.0, 1.0, 1.0])
normalize2 = TF.Normalize([0, 0, 0], config.std)
trans = TF.Compose([TF.ToTensor(), normalize1, normalize2, ])

revmean = [-x for x in config.mean]
revstd = [1.0 / x for x in config.std]
revnormalize1 = TF.Normalize([0.0, 0.0, 0.0], revstd)
revnormalize2 = TF.Normalize(revmean, [1.0, 1.0, 1.0])
revNormalize = TF.Compose([revnormalize1, revnormalize2])
revtrans = TF.Compose([revnormalize1, revnormalize2, TF.ToPILImage()])
to_img = TF.ToPILImage()

trainset = getattr(datas, config.trainset)(config.trainset_root, trans, config.train_size, config.train_crop_size, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size, shuffle=True, num_workers=8)
validationset = getattr(datas, config.validationset)(config.validationset_root, trans, config.validation_size, config.validation_crop_size, train=False)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=1, shuffle=False, num_workers=0)

# model
config.pwc_path = pwc_pth
model = getattr(models, config.model)(pwc_pth)

print('send model to npu, it might take minutes.')
model = model.npu()
vgg16_conv_4_3 = vgg16_conv_4_3.npu()
print('send model to npu done.')

model = nn.DataParallel(model)

# optimizer
params = list(model.module.refinenet.parameters()) + list(model.module.masknet.parameters())
optimizer = optim.Adam([{'params': params, 'initial_lr': config.init_learning_rate}], lr=config.init_learning_rate)

# scheduler to decrease learning rate by a factor of 10 at milestones.
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.3, last_epoch=72)
recorder = SummaryWriter(config.record_dir)

print('Everything prepared. Ready to train...')


# loss function
def lossfn(output, I1, I2, IT):
    It_warp = output

    recnLoss = F.l1_loss(It_warp, IT)
    prcpLoss = F.mse_loss(vgg16_conv_4_3(It_warp), vgg16_conv_4_3(IT))

    loss = 204 * recnLoss + 0.005 * prcpLoss

    return loss


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train():
    if config.train_continue:
        dict1 = torch.load(config.checkpoint, map_location=CALCULATE_DEVICE)
        model.load_state_dict(dict1['model_state_dict'])
        dict1['epoch'] = 72
    else:
        dict1 = {'loss': [], 'epoch': -1}
    if not os.path.exists(config.checkpoint_dir):
        os.mkdir(config.checkpoint_dir)
    start = time.time()
    for epoch in range(dict1['epoch'] + 1, config.epochs):
        print("Epoch: ", epoch)
        iLoss = 0
        for trainIndex, (trainData, t) in enumerate(trainloader, 0):
            # Get the input and the target from the training set
            frame0, frame1, frameT, frame2, frame3 = trainData
            d_s = time.time()
            I0 = frame0.npu()
            I1 = frame1.npu()
            I2 = frame2.npu()
            I3 = frame3.npu()
            IT = frameT.npu()
            t = t.view(t.size(0, ), 1, 1, 1).float().npu()
            d_e = time.time()
            optimizer.zero_grad()
            m_s = time.time()
            output = model(I0, I1, I2, I3, t)
            loss = lossfn(output, I1, I2, IT)
            m_e = time.time()
            loss.backward()
            l_e = time.time()
            optimizer.step()
            iLoss += loss.item()
            end = time.time()
            print("Iterations: %4d/%4d TrainExecTime: %0.1f(data=%0.1f, forward=%0.1f, loss=%0.1f) Loss=%f LearningRate: %f" % (trainIndex, len(trainloader), end - start, d_e - d_s, m_e - m_s, l_e - m_e, loss.item(), get_lr(optimizer)))
            start = time.time()
            itr = trainIndex + epoch * (len(trainloader))
            recorder.add_scalars('Loss', {'trainLoss': loss.item()}, itr)
            recorder.add_scalars('LearningRate', {'train': get_lr(optimizer)}, itr)
        # custom save
        recorder.add_scalars('EpochLoss', {'trainLoss': iLoss / len(trainloader)}, epoch)
        dict1 = {
            'Detail': "Quadratic video interpolation.",
            'epoch': epoch,
            'timestamp': datetime.datetime.now(),
            'trainBatchSz': config.train_batch_size,
            'validationBatchSz': 1,
            'learningRate': get_lr(optimizer),
            'loss': iLoss / len(trainloader),
            'valLoss': -1,
            'valPSNR': -1,
            'model_state_dict': model.state_dict(),
        }
        torch.save(dict1, config.checkpoint_dir + "/model" + str(epoch) + ".ckpt")
        scheduler.step()


train()
