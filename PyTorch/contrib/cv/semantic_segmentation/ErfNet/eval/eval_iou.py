# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
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
import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage
from torch.optim import Adam

from dataset import cityscapes
from erfnet import ERFNet
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry
from apex import amp

NUM_CHANNELS = 3
NUM_CLASSES = 20

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   #ignore label to 19
])

def init_process_group(proc_rank, world_size, device_type="npu", port="29588", dist_backend="hccl"):
    """Initializes the default process group."""

    # Initialize the process group
    print("==================================")    
    print('Begin init_process_group')
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = port
    if device_type == "npu":
        torch.distributed.init_process_group(
            backend=dist_backend,
            world_size=world_size,
            rank=proc_rank
        )
    elif device_type == "gpu":
        torch.distributed.init_process_group(
            backend=dist_backend,
            init_method="tcp://{}:{}".format("127.0.0.1", port),
            world_size=world_size,
            rank=proc_rank
        )        

    print("==================================")
    print("Done init_process_group")

    # Set the GPU to use
    #torch.cuda.set_device(proc_rank)
    if device_type == "npu":
        torch.npu.set_device(proc_rank)
    elif device_type == "gpu":
        torch.cuda.set_device(proc_rank)
    print('Done set device', device_type, dist_backend, world_size, proc_rank)

def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    if args.num_gpus > 1:
        init_process_group(proc_rank=args.rank_id, world_size=args.num_gpus, device_type=args.device)
    elif args.device == "npu":
        torch.npu.set_device(0)
    elif args.device == "gpu":
        torch.cuda.set_device(0)
        
    from erfnet_imagenet import ERFNet as ERFNet_imagenet
    pretrainedEnc = ERFNet_imagenet(1000)
    pretrainedEnc = next(pretrainedEnc.children()).encoder
    model = ERFNet(NUM_CLASSES, encoder=pretrainedEnc)
    model = model.npu()
    
    if args.device == "npu":  
        cur_device = torch.npu.current_device()
    elif args.device == "gpu":
        cur_device = torch.cuda.current_device()
    print('cur_device: ', cur_device)

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")
     
    if args.num_gpus > 1:
        #Make model replica operate on the current device
        ddp = torch.nn.parallel.DistributedDataParallel
        model = ddp(model, device_ids=[cur_device],  broadcast_buffers=False, find_unused_parameters=True)

    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")


    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)


    iouEvalVal = iouEval(NUM_CLASSES)

    start = time.time()
    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        images = images.npu()
        labels = labels.npu()

        with torch.no_grad():
            outputs = model(images)  
        
        final_outputs = outputs.max(1)[1].unsqueeze(1)
        iouEvalVal.addBatch(final_outputs, labels)

        filenameSave = filename[0].split("leftImg8bit/")[1] 
        if (args.num_gpus == 1 or (args.num_gpus > 1
                and args.rank_id % args.num_gpus == 0)):
            print(step, filenameSave)

    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    if (args.num_gpus == 1 or (args.num_gpus > 1
        and args.rank_id % args.num_gpus == 0)):
        print("---------------------------------------")
        print("Took ", time.time()-start, "seconds")
        print("=======================================")
        print("Per-Class IoU:")
        print(iou_classes_str[0], "Road")
        print(iou_classes_str[1], "sidewalk")
        print(iou_classes_str[2], "building")
        print(iou_classes_str[3], "wall")
        print(iou_classes_str[4], "fence")
        print(iou_classes_str[5], "pole")
        print(iou_classes_str[6], "traffic light")
        print(iou_classes_str[7], "traffic sign")
        print(iou_classes_str[8], "vegetation")
        print(iou_classes_str[9], "terrain")
        print(iou_classes_str[10], "sky")
        print(iou_classes_str[11], "person")
        print(iou_classes_str[12], "rider")
        print(iou_classes_str[13], "car")
        print(iou_classes_str[14], "truck")
        print(iou_classes_str[15], "bus")
        print(iou_classes_str[16], "train")
        print(iou_classes_str[17], "motorcycle")
        print(iou_classes_str[18], "bicycle")
        print("=======================================")
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    if (args.num_gpus == 1 or (args.num_gpus > 1
        and args.rank_id % args.num_gpus == 0)):
        print ("MEAN IoU: ", iouStr, "%")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    # apex setting
    parser.add_argument('--amp', default=False, action='store_true',
                        help='use amp to train the model')
    parser.add_argument('--opt-level', default="O2", type=str, help='apex optimize level')
    parser.add_argument('--loss-scale-value', default='128', type=int, help='static loss scale value')

    # device setting
    parser.add_argument("--device", default="npu", type=str)    
    parser.add_argument("--rank_id", dest="rank_id", default=0, type=int)
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--addr", default="127.0.0.1", type=str)
    parser.add_argument("--port", default="29588", type=str)
    parser.add_argument("--dist_backend", default="hccl", type=str)
    main(parser.parse_args())
