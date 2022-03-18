# Main code for training ERFNet model in Cityscapes dataset
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
import random
import time
import numpy as np
import torch
import math

from PIL import Image, ImageOps
from argparse import ArgumentParser

import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data.distributed import DistributedSampler

from dataset import VOC12,cityscapes
from transform import Relabel, ToLabel, Colorize

import importlib
from iouEval import iouEval, getColorEntry

from shutil import copyfile

from apex import amp
from apex.optimizers import NpuFusedAdam

NUM_CHANNELS = 3
NUM_CLASSES = 20 #pascal=22, cityscapes=20

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()    
    
#Augmentations - different function implemented to perform random augments on both image and target
class MyCoTransform(object):
    def __init__(self, enc, augment=True, height=512):
        self.enc=enc
        self.augment = augment
        self.height = height
        pass
    def __call__(self, input, target):
        # do something to both images
        input =  Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)

        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            
            #Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 255
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))   

        input = ToTensor()(input)
        if (self.enc):
            target = Resize(int(self.height/8), Image.NEAREST)(target)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)


def load_my_pretrained_state_dict(model, state_dict, is_finetune):  #custom function to load model when not all dict elements
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            if name.startswith("module."):
                new_name = name.split("module.")[-1]
                if new_name.startswith("decoder.output_conv.") and is_finetune:
                    print(name, " not loaded")
                    continue
                own_state[new_name].copy_(param)
            else:
                print(name, " not loaded")
                continue
        else:
            own_state[name].copy_(param)
    return model
    

def profiling(loader, model, criterion, optimizer, args):
    # switch to train mode
    model.train()

    def update(model, images, target, optimizer):
        output = model(images, only_encode=False)
        loss = criterion(output, target[:, 0])
        optimizer.zero_grad()
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()      
        optimizer.step()

    for step, (images, target) in enumerate(loader):
        if 'npu' in args.device:
            target = target.to(torch.int32)

        if 'npu' in args.device:
            images = images.npu()
            target = target.npu()
            
        if step < 5:
            update(model, images, target, optimizer)
        else:
            if args.device == 'npu':
                with torch.autograd.profiler.profile(use_npu=True) as prof:
                    update(model, images, target, optimizer)
            elif args.device == "gpu":
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    update(model, images, target, optimizer)
            break

    prof.export_chrome_trace("erfnet_output.prof")


def train(args, model, enc=False, profile=False):
    best_acc = 0
    savedir = f'save/{args.savedir}'
    #TODO: calculate weights by processing dataset histogram (now its being set by hand from the torch values)
    #create a loder to run all images and calculate histogram of labels, then create weight array using class balancing

    weight = torch.ones(NUM_CLASSES)
    if (enc):
        weight[0] = 2.3653597831726	
        weight[1] = 4.4237880706787	
        weight[2] = 2.9691488742828	
        weight[3] = 5.3442072868347	
        weight[4] = 5.2983593940735	
        weight[5] = 5.2275490760803	
        weight[6] = 5.4394111633301	
        weight[7] = 5.3659925460815	
        weight[8] = 3.4170460700989	
        weight[9] = 5.2414722442627	
        weight[10] = 4.7376127243042	
        weight[11] = 5.2286224365234	
        weight[12] = 5.455126285553	
        weight[13] = 4.3019247055054	
        weight[14] = 5.4264230728149	
        weight[15] = 5.4331531524658	
        weight[16] = 5.433765411377	
        weight[17] = 5.4631009101868	
        weight[18] = 5.3947434425354
    else:
        weight[0] = 2.8149201869965	
        weight[1] = 6.9850029945374	
        weight[2] = 3.7890393733978	
        weight[3] = 9.9428062438965	
        weight[4] = 9.7702074050903	
        weight[5] = 9.5110931396484	
        weight[6] = 10.311357498169	
        weight[7] = 10.026463508606	
        weight[8] = 4.6323022842407	
        weight[9] = 9.5608062744141	
        weight[10] = 7.8698215484619	
        weight[11] = 9.5168733596802	
        weight[12] = 10.373730659485	
        weight[13] = 6.6616044044495	
        weight[14] = 10.260489463806	
        weight[15] = 10.287888526917	
        weight[16] = 10.289801597595	
        weight[17] = 10.405355453491	
        weight[18] = 10.138095855713	
    weight[19] = 0
    weight = weight.npu()
    criterion = CrossEntropyLoss2d(weight).npu()
    print(type(criterion))

    #TODO: reduce memory in first gpu: https://discuss.pytorch.org/t/multi-gpu-training-memory-usage-in-balance/4163/4        #https://github.com/pytorch/pytorch/issues/1893

    #optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=2e-4)     ## scheduler 1
    optimizer = NpuFusedAdam(model.parameters(), args.lr, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)      ## scheduler 2

    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale_value, combine_grad=True)

    if args.device == "npu":  
        cur_device = torch.npu.current_device()
    elif args.device == "gpu":
        cur_device = torch.cuda.current_device()
    print('cur_device: ', cur_device)
  
    if args.num_gpus > 1:
        #Make model replica operate on the current device
        ddp = torch.nn.parallel.DistributedDataParallel
        model = ddp(model, device_ids=[cur_device],  broadcast_buffers=False, find_unused_parameters=True)

    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5) # set up scheduler     ## scheduler 1
    lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9)  ## scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)                             ## scheduler 2

    if args.resume:
        #Must load weights, optimizer, epoch and best value. 
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'

        assert os.path.exists(filenameCheckpoint), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(filenameCheckpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        if args.amp:
            amp.load_state_dict(checkpoint['amp'])
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))

    assert os.path.exists(args.datadir), "Error: datadir (dataset directory) could not be loaded"
    co_transform = MyCoTransform(enc, augment=True, height=args.height)#1024)
    co_transform_val = MyCoTransform(enc, augment=False, height=args.height)#1024)
    dataset_train = cityscapes(args.datadir, co_transform, 'train')
    train_sampler = DistributedSampler(dataset_train) if args.num_gpus > 1 else None
    dataset_val = cityscapes(args.datadir, co_transform_val, 'val')
    val_sampler = DistributedSampler(dataset_val) if args.num_gpus > 1 else None
    loader = DataLoader(
        dataset_train, 
        num_workers=int(args.num_workers / args.num_gpus), 
        batch_size=int(args.batch_size / args.num_gpus),
        shuffle=(False if train_sampler else True),
        sampler=train_sampler)
    loader_val = DataLoader(
        dataset_val, 
        num_workers=int(args.num_workers / args.num_gpus), 
        batch_size=int(args.batch_size / args.num_gpus),
        shuffle=False)

    if (args.num_gpus == 1 or (args.num_gpus > 1
        and args.rank_id % args.num_gpus == 0)):
        if (enc):
            automated_log_path = savedir + "/automated_log_encoder.txt"
            modeltxtpath = savedir + "/model_encoder.txt"
        else:
            automated_log_path = savedir + "/automated_log.txt"
            modeltxtpath = savedir + "/model.txt"    
        if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
            with open(automated_log_path, "a") as myfile:
                myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")
        with open(modeltxtpath, "w") as myfile:
            myfile.write(str(model))

    if args.profile:
        print("----- DO PROFILE -----")
        profiling(loader, model, criterion, optimizer, args)
        print("----- Done PROFILE -----")
        return

    start_epoch = 1
    for epoch in range(start_epoch, args.num_epochs+1):
        if (args.num_gpus == 1 or (args.num_gpus > 1
            and args.rank_id % args.num_gpus == 0)):
            print("----- TRAINING - EPOCH", epoch, "-----")

        scheduler.step(epoch)    ## scheduler 2
        if args.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        epoch_loss = []
        time_train = []
        fps_train = []

        doIouTrain = args.iouTrain   
        doIouVal =  args.iouVal      

        if (doIouTrain):
            iouEvalTrain = iouEval(NUM_CLASSES)

        usedLr = 0
        if (args.num_gpus == 1 or (args.num_gpus > 1
            and args.rank_id % args.num_gpus == 0)):
            for param_group in optimizer.param_groups:
                print("LEARNING RATE: ", param_group['lr'])
                usedLr = float(param_group['lr'])

        model.train()

        for step, (inputs, targets) in enumerate(loader):
            start_time = time.time()
            
            inputs = inputs.npu()
            targets = targets.to(torch.int32)
            targets = targets.npu()

            outputs = model(inputs, only_encode=enc)

            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)
            if step > 5:
                fps_train.append(time.time() - start_time)
                
            final_outputs = outputs.max(1)[1].unsqueeze(1)
            final_outputs = final_outputs.to(torch.int32)
            if (doIouTrain):
                #start_time_iou = time.time()
                iouEvalTrain.addBatch(final_outputs, targets)
                #print ("Time to add confusion matrix: ", time.time() - start_time_iou)      

            if (args.steps_loss > 0 and step % args.steps_loss == 0) and (args.num_gpus == 1 or (args.num_gpus > 1
                and args.rank_id % args.num_gpus == 0)):
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

        if (args.num_gpus == 1 or (args.num_gpus > 1
                and args.rank_id % args.num_gpus == 0)):
            fps = args.batch_size / (sum(fps_train) / len(fps_train))
            print("now epoch : ", epoch, " fps : %.2f" % fps)

        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        
        iouTrain = 0
        if (doIouTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            iouStr = getColorEntry(iouTrain)+'{:0.2f}'.format(iouTrain*100) + '\033[0m'
            if (args.num_gpus == 1 or (args.num_gpus > 1
                and args.rank_id % args.num_gpus == 0)):
                print ("EPOCH IoU on TRAIN set: ", iouStr, "%")  

        #Validate on 500 val images after each epoch of training
        if (args.num_gpus == 1 or (args.num_gpus > 1
            and args.rank_id % args.num_gpus == 0)):
            print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        epoch_loss_val = []
        time_val = []

        if (doIouVal):
            iouEvalVal = iouEval(NUM_CLASSES)

        for step, (inputs, targets) in enumerate(loader_val):
            start_time = time.time()

            inputs = inputs.npu()
            targets = targets.npu()

            with torch.no_grad():
                outputs = model(inputs, only_encode=enc) 

            loss = criterion(outputs, targets[:, 0])
            epoch_loss_val.append(loss.item())
            time_val.append(time.time() - start_time)

            final_outputs = outputs.max(1)[1].unsqueeze(1)
            #Add batch to calculate TP, FP and FN for iou estimation
            if (doIouVal):
                #start_time_iou = time.time()
                iouEvalVal.addBatch(final_outputs, targets)
                #print ("Time to add confusion matrix: ", time.time() - start_time_iou)

            if (args.steps_loss > 0 and step % args.steps_loss == 0) and (args.num_gpus == 1 or (args.num_gpus > 1
                and args.rank_id % args.num_gpus == 0)):
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
                       
        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
        #scheduler.step(average_epoch_loss_val, epoch)  ## scheduler 1   # update lr if needed

        iouVal = 0
        if (doIouVal):
            iouVal, iou_classes = iouEvalVal.getIoU()
            iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
            if (args.num_gpus == 1 or (args.num_gpus > 1
                and args.rank_id % args.num_gpus == 0)):
                print ("EPOCH IoU on VAL set: ", iouStr, "%") 
           

        # remember best valIoU and save checkpoint (save checkpoint)
        if iouVal == 0:
            current_acc = -average_epoch_loss_val
        else:
            current_acc = iouVal 
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)
        if (args.num_gpus == 1 or (args.num_gpus > 1
            and args.rank_id % args.num_gpus == 0)):
            if enc:
                filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
                filenameBest = savedir + '/model_best_enc.pth.tar'    
            else:
                filenameCheckpoint = savedir + '/checkpoint.pth.tar'
                filenameBest = savedir + '/model_best.pth.tar'
            if args.amp:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': str(model),
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                    'amp' : amp.state_dict(),
                }, is_best, filenameCheckpoint, filenameBest)
            else:   
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': str(model),
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, filenameCheckpoint, filenameBest)            

        #SAVE MODEL AFTER EPOCH (save model)
        if (args.num_gpus == 1 or (args.num_gpus > 1
            and args.rank_id % args.num_gpus == 0)):
            if (enc):
                filename = f'{savedir}/model_encoder-{epoch:03}.pth'
                filenamebest = f'{savedir}/model_encoder_best.pth'
            else:
                filename = f'{savedir}/model-{epoch:03}.pth'
                filenamebest = f'{savedir}/model_best.pth'

            if args.epochs_save > 0 and step > 0 and step % args.epochs_save == 0:
                torch.save(model.state_dict(), filename)
                print(f'save: {filename} (epoch: {epoch})')
            if (is_best):
                torch.save(model.state_dict(), filenamebest)
                print(f'save: {filenamebest} (epoch: {epoch})')
                if (not enc):
                    with open(savedir + "/best.txt", "w") as myfile:
                        myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))   
                else:
                    with open(savedir + "/best_encoder.txt", "w") as myfile:
                        myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))           

            #SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
            #Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
            with open(automated_log_path, "a") as myfile:
                myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr ))
    
    return(model)   #return model (convenience for encoder-decoder training)


def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)


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
    savedir = f'save/{args.savedir}'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    SEED= 5
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = True
    
    if args.num_gpus > 1:
        init_process_group(proc_rank=args.rank_id, world_size=args.num_gpus, device_type=args.device)
    elif args.device == "npu":
        torch.npu.set_device(0)
    elif args.device == "gpu":
        torch.cuda.set_device(0)

    #Load Model
    #assert os.path.exists(args.model + ".py"), "Error: model definition not found"
    model_file = importlib.import_module(args.model)
    model = model_file.Net(NUM_CLASSES)
    model = model.npu()

    copyfile("train" + '/' + args.model + ".py", savedir + '/' + args.model + ".py")
    
    if args.state:
        #if args.state is provided then load this state for training
        #Note: this only loads initialized weights. If you want to resume a training use "--resume" option!!
        def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict keys are there
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                     continue
                own_state[name].copy_(param)
            return model

        #print(torch.load(args.state))
        model = load_my_state_dict(model, torch.load(args.state))

    #train(args, model)
    if (not args.decoder):
        print("========== ENCODER TRAINING ===========")
        model = train(args, model, True) #Train encoder

    #CAREFUL: for some reason, after training encoder alone, the decoder gets weights=0. 
    #We must reinit decoder weights or reload network passing only encoder in order to train decoder
    print("========== DECODER TRAINING ===========")
    if (not args.state):
        if args.pretrainedEncoder:
            print("Loading encoder pretrained in imagenet")
            from erfnet_imagenet import ERFNet as ERFNet_imagenet
            pretrainedEnc = ERFNet_imagenet(1000)
            pretrainedEnc = load_my_pretrained_state_dict(pretrainedEnc, torch.load(args.pretrainedEncoder, map_location='cpu')['state_dict'], args.finetune)
            pretrainedEnc = next(pretrainedEnc.children()).encoder
        else:
            if args.decoder or args.num_gpus == 1:
                pretrainedEnc = next(model.children())
            else:
                pretrainedEnc = next(model.children()).encoder
        if args.finetune:
            model = model_file.Net(args.fnum, encoder=pretrainedEnc)
            model = load_my_pretrained_state_dict(model, torch.load(args.pretrainedDecoder, map_location='cpu')['state_dict'], args.finetune)
        else:
            model = model_file.Net(NUM_CLASSES, encoder=pretrainedEnc)  #Add decoder to encoder
        model = model.npu()
        #When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
    model = train(args, model, False)   #Train decoder

    print("========== TRAINING FINISHED ===========")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="erfnet")
    parser.add_argument('--savedir', type=str, default="erfnet_training1")
    parser.add_argument('--state', action='store_true', default=False)
    parser.add_argument('--decoder', action='store_true', default=False)
    parser.add_argument('--pretrainedEncoder') #, default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--pretrainedDecoder')
    parser.add_argument("--finetune", action='store_true', default=False)
    parser.add_argument('--fnum', type=int, default=20)    

    parser.add_argument('--datadir', type=str, default="/home/cityscapes/")
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--epochs-save', type=int, default=0)    #You can use this value to save model every X epochs
    parser.add_argument('--lr', '--learning_rate', default=5e-4, type=float,
                        metavar='LR', help='initial learning rate')

    parser.add_argument('--iouTrain', action='store_true', default=False) #recommended: False (takes more time to train otherwise)
    parser.add_argument('--iouVal', action='store_true', default=True)  
    parser.add_argument('--resume', action='store_true')    #Use this flag to load last checkpoint for training  

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

    parser.add_argument("--profile", default=0, type=int)
    main(parser.parse_args())
