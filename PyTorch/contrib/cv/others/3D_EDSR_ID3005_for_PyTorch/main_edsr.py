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
# main code for 3D EDSR training
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torch.npu
from edsr_x3_3d import EDSR
from load_data import DatasetFromFolder
import math
import time
from option import args


def main():
    cudnn.benchmark = True
    print("Random Seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.use_npu:
        print("===> Use NPU")
        device = torch.device("npu:{}".format(args.npu))
        print(device)
        torch.npu.set_device(device)
    else:
        print("===> Use GPU")
        device = torch.device("cuda:{}".format(args.cuda))

    print("===> Loading datasets")
    # load train/test data
    train_set = DatasetFromFolder(args.dir_train_data)
    test_set = DatasetFromFolder(args.dir_test_data)
    training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batch_size,
                                      shuffle=True)
    test_data_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    print("===> Building model")
    model = EDSR()
    criterion = nn.L1Loss(reduction='mean')
    print("===> Setting GPU/NPU")
    model = model.to(device)
    criterion = criterion.to(device)

    print("===> Setting Optimizer")
    print("===> Training")
    for epoch in range(1, args.epochs + 1):
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas)
        lr = adjust_learning_rate(epoch - 1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Epoch={} Train, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
        train(training_data_loader, optimizer, model, criterion, epoch, device)
        test(test_data_loader, model, criterion, epoch, device)
    torch.save(model.state_dict(), './3D_EDSR.pt')


def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = args.lr * (args.gamma ** (epoch // args.step))
    return lr


def train(training_data_loader, optimizer, model, criterion, epoch, device):
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)  # requires_grad=False
        input = input.to(device)
        target = target.to(device)
        t0 = time.time()
        sr = model(input)
        loss = criterion(sr, target)
        loss = loss.to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t1 = time.time()
        mse = torch.mean((sr - target) ** 2)
        psnr = 10 * math.log10(1.0 / torch.mean(mse))
        if iteration % 10 == 0:
            print("===> Epoch[{}]({}/{}): Train Loss:{:.10f} psnr:{:.10f}".format( \
                epoch, iteration, len(training_data_loader), loss.item(), psnr))
            print('===> Timer:%.4f' % (t1 - t0))


def test(test_data_loader, model, criterion, epoch, device):
    model.eval()
    loss = 0
    psnr = 0
    t1 = time.time()
    count = 0
    for iteration, batch in enumerate(test_data_loader, 1):
        with torch.no_grad():
            input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
            input = input.to(device)
            target = target.to(device)
            sr = model(input)
            loss_tem = criterion(sr, target)
            loss_tem = loss_tem.to(device)
            loss += loss_tem.item()
            mse = torch.mean((sr - target) ** 2)
            psnr_tem = 10 * math.log10(1.0 / torch.mean(mse))
            psnr += psnr_tem
            count += 1

    tloss = loss / count
    tpsnr = psnr / count
    t2 = time.time()
    print("Epoch[{}]: Test Loss:{:.10f} psnr:{:.10f} Timer:{:.4f}".format( \
        epoch, tloss, tpsnr, (t2 - t1)))
    if epoch == args.epochs:
        print("Final Accuracy accuracy psnr: {:.10f} ".format(tpsnr))

if __name__ == "__main__":
    main()
