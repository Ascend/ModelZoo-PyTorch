# MIT License
#
# Copyright (c) 2020 xxx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ============================================================================
#
# Copyright 2021 Huawei Technologies Co., Ltd
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
if torch.__version__ >= "1.8.1":
    import torch_npu
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import models
import losses
import utils_train
import change_dataset_np
import argparse
import os
import apex

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=50, type=int,
                        help='Total number of traning epochs to perform.')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='Total number classes.')
    parser.add_argument('--batch_size', default=20, type=int,
                        help='The batch size for train.')
    parser.add_argument('--img_size', default=224, type=int,
                        help='The image size of dataset.')
    parser.add_argument('--base_lr', default=1e-4, type=float,
                        help='The learning rate for train.')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='The local rank for distribute training.')
    parser.add_argument("--fp16", default=False, action="store_true",
                        help="whether to use 16-bit(mixed) precision instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16:['O0','O1','O2','O3']")
    parser.add_argument("--output_dir", type=str, default="./", required=True,
                        help="the output directory")
    parser.add_argument("--data_dir", type=str, default="./", required=True,
                        help="the dataset directory")

    args = parser.parse_args()

    if args.local_rank == -1:
        device = torch.device("npu", int(os.environ['ASCEND_DEVICE_ID']))
        torch.npu.set_device(device)
        num_gpu = 1
    else:
        device = torch.device("npu", args.local_rank)
        torch.npu.set_device(device)
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        torch.distributed.init_process_group(backend="hccl", world_size=world_size, rank=rank)
        num_gpu = world_size

    args.base_lr *= num_gpu
    print('Number of GPUs Available:', num_gpu)

    train_pickle_file = os.path.join(args.data_dir, 'VL-CMU-CD/change_dataset_train.pkl')
    val_pickle_file = os.path.join(args.data_dir, 'VL-CMU-CD/change_dataset_val.pkl')

    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data_transforms = {
        'train': transforms.Compose([
            #transforms.RandomResizedCrop(img_size),
            #transforms.RandomHorizontalFlip(),
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            #helper_augmentations.SwapReferenceTest(),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
            #helper_augmentations.JitterGamma(),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    # Create training and validation datasets
    train_dataset = change_dataset_np.ChangeDatasetNumpy(train_pickle_file, data_transforms['train'])
    val_dataset = change_dataset_np.ChangeDatasetNumpy(val_pickle_file, data_transforms['val'])

    # Create training and validation dataloaders
    if args.local_rank == -1:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   pin_memory=True,
                                                   num_workers=16)
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   sampler=sampler,
                                                   batch_size=args.batch_size,
                                                   drop_last=True,
                                                   pin_memory=True,
                                                   num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             drop_last=True,
                                             pin_memory=True,
                                             num_workers=16)
    #dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=8) for x in ['train', 'val']}
    dataloaders_dict = {'train': train_loader, 'val': val_loader}

    # Default directory "runs"
    writer = SummaryWriter()

    img_reference_dummy = torch.randn(1, 3, args.img_size, args.img_size)
    img_test_dummy = torch.randn(1, 3, args.img_size, args.img_size)
    change_net = models.ChangeNet(num_classes=args.num_classes, model_dir=args.data_dir)

    # Add on Tensorboard the Model Graph
    writer.add_graph(change_net, [img_reference_dummy, img_test_dummy])

    change_net = change_net.to(device)

    #criterion = nn.CrossEntropyLoss()
    # If there are more than 2 classes the alpha need to be a list
    criterion = losses.FocalLoss(gamma=2.0, alpha=[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
    if args.fp16:
        optimizer = apex.optimizers.NpuFusedAdam(change_net.parameters(), lr=args.base_lr)
    else:
        optimizer = optim.Adam(change_net.parameters(), lr=args.base_lr)
    sc_plt = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3,
                                                        verbose=True if args.local_rank in [0, -1] else False)

    if args.fp16:
        change_net, optimizer = apex.amp.initialize(change_net, optimizer, opt_level=args.fp16_opt_level,
                                                    loss_scale=65536,
                                                    combine_grad=True)

    if args.local_rank != -1:
        change_net = nn.parallel.DistributedDataParallel(change_net, device_ids=[args.local_rank],
                                                         find_unused_parameters=True)

    utils_train.train_model(change_net, dataloaders_dict, criterion, optimizer, sc_plt, writer, device, args)


if __name__ == "__main__":
    main()





