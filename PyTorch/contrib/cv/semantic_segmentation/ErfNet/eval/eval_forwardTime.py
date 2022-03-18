# Code to evaluate forward pass time in Pytorch
# Sept 2017
# Eduardo Romera
#######################
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
import os
import numpy as np
import torch
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable

from erfnet_nobn import ERFNet
from transform import Relabel, ToLabel, Colorize

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def main(args):
    model = ERFNet(19)
    if (not args.cpu):
        model = model.cuda()#.half()	#HALF seems to be doing slower for some reason
    #model = torch.nn.DataParallel(model).cuda()

    model.eval()


    images = torch.randn(args.batch_size, args.num_channels, args.height, args.width)

    if (not args.cpu):
        images = images.cuda()#.half()

    time_train = []

    i=0

    while(True):
    #for step, (images, labels, filename, filenameGt) in enumerate(loader):

        start_time = time.time()

        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs)

        #preds = outputs.cpu()
        if (not args.cpu):
            torch.cuda.synchronize()    #wait for cuda to finish (cuda is asynchronous!)

        if i!=0:    #first run always takes some time for setup
            fwt = time.time() - start_time
            time_train.append(fwt)
            print ("Forward time per img (b=%d): %.3f (Mean: %.3f)" % (args.batch_size, fwt/args.batch_size, sum(time_train) / len(time_train) / args.batch_size))
        
        time.sleep(1)   #to avoid overheating the GPU too much
        i+=1

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())
