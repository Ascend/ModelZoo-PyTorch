# Copyright 2021 Huawei Technologies Co., Ltd
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

from __future__ import print_function, division
from apex import amp
from apex.optimizers import NpuFusedSGD
import torch
if torch.__version__ >= "1.8.1":
    import torch_npu
else:
    import torch.npu
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms, datasets, models
import os
import cv2
import time
import torch.utils.data.distributed
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel

device = None
args = None 
log_file = None

def parse_args():
    parser = argparse.ArgumentParser(description='3D Attention Net')
    parser.add_argument('--device_type', type=str)
    parser.add_argument('--device_id', type=int)
    parser.add_argument('--device_num', type=int)
    parser.add_argument('--total_epochs', type=int)
    parser.add_argument('--is_train', type=str)
    parser.add_argument('--is_pretrain', type=str)
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--dist_url', type=str)
    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--test_batch_size', type=int)
    args = parser.parse_args()
    return args

# for test
def test(model, test_loader):
    # Test
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    for images, labels in test_loader:
        images = Variable(images.to(device))
        labels = Variable(labels.to(device))
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        c = (predicted == labels.data).squeeze()
        for i in range(len(labels.data)):
            label = labels.data[i]
            class_correct[label] += c[i]
            class_total[label] += 1
    if args.device_id == 0:
         print(f"Accuracy of the model(on device: {args.device_id}) on the test images: {100 * float(correct) / total} %")
    write_log('Accuracy of the model on the test images: %d %%\n' % (100 * float(correct) / total))
    write_log(f'Accuracy of the model on the test images: {float(correct)/total} \n')
    return float(correct) / total

def write_log(output):
    if log_file is not None:
        log_file.write(output)


def main():
    global args
    global device
    args = parse_args()
    print(args)
    model_file = 'model_92_sgd.pkl'
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    lr = 0.1
    is_train = args.is_train == "True"
    is_pretrain = args.is_pretrain == "True"
    acc_best = 0
    total_epoch = args.total_epochs
    distribute = args.device_num > 1
    if(args.device_type == "GPU"):
        device = torch.device("cuda", args.device_id)
        if distribute:
            torch.cuda.set_device(args.device_id)
            torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url, world_size=args.device_num, rank=args.device_id)
    else:
        device = f"npu:{args.device_id}"
        if distribute:
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '49876'
            torch.npu.set_device(device)
            print("rank:",args.device_id)
            torch.distributed.init_process_group(backend="hccl", world_size=args.device_num, rank=args.device_id)
    
    # Image Preprocessing
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((32, 32), padding=4), 
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.CIFAR10(root='./data/', train=True, transform=transform, download=False)
    test_dataset = datasets.CIFAR10(root='./data/', train=False, transform=test_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distribute else None
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, \
                                                batch_size=train_batch_size, \
                                                shuffle=(train_sampler is None), \
                                                num_workers=8, \
                                                pin_memory=False, \
                                                sampler = train_sampler if is_train else None, \
                                                drop_last = True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)
    
    model = ResidualAttentionModel(args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = None
    if args.device_type == "GPU":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
    else:
        optimizer = NpuFusedSGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale="dynamic")
    if distribute:
        if args.device_type == "GPU":
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.device_id])
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.device_id], broadcast_buffers=False)
    global log_file
    if args.device_id == 0:
        log_file = open("train_log_file" if is_train else "test_log_file", mode="w+") 
    if is_train is True:
        if is_pretrain == True:
            base_weights = torch.load(model_file, map_location="cpu")
            print('Loading base network...')
            new_state_dict = OrderedDict()
            for k, v in base_weights.items():
                if(k[0: 7] == "module."):
                    name = k[7:]
                else:
                    name = k[0:]
                new_state_dict[name] = v 
            if "fc.weight" in new_state_dict:
                print("pop fc layer weight")
                new_state_dict.pop("fc.weight")
                new_state_dict.pop("fc.bias")
            model.load_state_dict(new_state_dict, strict=False)

        # Training
        total_tims = 0
        total_samples = 0
        for epoch in range(total_epoch):
            model.train()
            tims = time.time()
            epoch_samples = 0
            if train_sampler is not None: # is distributed
                train_sampler.set_epoch(epoch)
            for i, (images, labels) in enumerate(train_loader):
                epoch_samples += images.shape[0]
                if i == 5:
                    tims = time.time()
                images = Variable(images.to(device))
                labels = Variable(labels.to(device))
                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
    
                if (i+1) % 20 == 0 and args.device_id == 0:
                    print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" %(epoch+1, total_epoch, i+1, len(train_loader), loss.item()))
                    write_log("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f \n" %(epoch+1, total_epoch, i+1, len(train_loader), loss.item()))
            total_tims += time.time() - tims
            total_samples += epoch_samples
            if args.device_id == 0:
                print(f'the epoch {epoch+1} takes time:',time.time()-tims)
                print(f"epoch {epoch+1} FPS: {(epoch_samples - 5 * train_batch_size)* args.device_num / (time.time()-tims)}")
                print('evaluate test set:')
            write_log(f'the epoch {epoch+1} takes time: {time.time()-tims} \n')
            write_log(f"epoch {epoch+1} FPS: {(epoch_samples - 5 * train_batch_size)* args.device_num / (time.time()-tims)} \n")
            acc = test(model, test_loader)
            if acc > acc_best:
                acc_best = acc
                print('current best acc,', acc_best)
                if args.device_id == 0:
                    torch.save(model.state_dict(), model_file)
            # Decaying Learning Rate
            if (epoch+1) / float(total_epoch) == 0.3 or (epoch+1) / float(total_epoch) == 0.6 or (epoch+1) / float(total_epoch) == 0.9:
                lr /= 10
                print('reset learning rate to:', lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    print(param_group['lr'])
        # Save the Model
        if args.device_id == 0:
            torch.save(model.state_dict(), 'last_model_92_sgd.pkl')
    elif args.device_id == 0:
        base_weights = torch.load(model_file, map_location="cpu")
        print('Loading base network...')
        new_state_dict = OrderedDict()
        for k, v in base_weights.items():
            if(k[0: 7] == "module."):
                name = k[7:]
            else:
                name = k[0:]
            new_state_dict[name] = v 
        model.load_state_dict(new_state_dict)
        test(model, test_loader)

if __name__ == "__main__":
    main()
