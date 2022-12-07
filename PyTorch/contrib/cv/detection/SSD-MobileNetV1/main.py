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
# ============================================================================
import argparse
import os
import logging
import sys
import itertools
import time

import torch
if torch.__version__ >= '1.8':
    import torch_npu
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, LambdaLR
import apex
from apex import amp

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.datasets.voc_dataset import VOCDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')

parser.add_argument('--datasets', type=str, help='Dataset directory path')
parser.add_argument('--validation_dataset', help='Dataset directory path')
parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")
parser.add_argument('--freeze_base_net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze_net', action='store_true',
                    help="Freeze all the layers except the prediction head.")
parser.add_argument('--addr', default='127.0.0.1', type=str, help='master addr')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--base_net_lr', default=None, type=float,
                    help='initial learning rate for base net.')
parser.add_argument('--extra_layers_lr', default=None, type=float,
                    help='initial learning rate for the layers not in base net and prediction heads.')


# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net',
                    help='Pretrained base model')
parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--class_num',default=20,type=int, help='nums of class')
# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,100", type=str,
                    help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t_max', default=120, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=120, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=5, type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps', default=100, type=int,
                    help='Set the debug log output frequency.')

parser.add_argument('--checkpoint_folder', default='models/',
                    help='Directory for saving checkpoint models')

parser.add_argument('--distributed', default=False, type=str2bool,
                    help='Use 1/8p to train model')
parser.add_argument('--dist_backend', default='hccl', type=str,
                    help='gpu or npu')
parser.add_argument('--world_size', default=1, type=int,
                    help='nums of processes')      
parser.add_argument('--local_rank', default=0, type=int,
                    help='node rank/npu for distributed training')      
parser.add_argument("--amp", default=True, type=str2bool, help='if use amp')  
parser.add_argument('--opt_level', default='O1', type=str, help='apex optimize level')
parser.add_argument('--loss_scale_value', default=128.0, type=float, help='static loss scale value')
parser.add_argument('--npu', default=0, type=int,
                    help='use which npu to train')  
parser.add_argument('--prof', default=False, type=str2bool,
                    help='if save prof')
parser.add_argument('--main_rank', default=0, type=int,
                    help='node rank/npu for distributed training') 

parser.add_argument('--warm_up', default=False, action='store_true', help='use warm_up or not')
parser.add_argument('--warm_up_epochs', default=5, type=int, help='warm up epochs')

def train(args, loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        if i >= 5 and args.local_rank == args.main_rank:
            start = time.time()
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        if args.prof:
            with torch.autograd.profiler.profile(use_npu=True) as prof: 
                optimizer.zero_grad()
                confidence, locations = net(images)
                regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
                loss = regression_loss + classification_loss
                # 混合精度修改 
                if args.amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
            if i == 20:
                print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                prof.export_chrome_trace("output.prof") # "output.prof"为输出文件地址
        else:
            optimizer.zero_grad()
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
            loss = regression_loss + classification_loss
            # 混合精度修改 
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

        # calculate mAP


        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0  and args.local_rank == args.main_rank:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0
        if i >=5 and args.local_rank == args.main_rank:
            end = time.time()
            fps = (args.batch_size)*(args.world_size) / (end - start)
            logging.info(f"Epoch : {epoch}, npu : {args.local_rank}, step: {i}, FPS is : {fps}")
            start = time.time()

def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '29688'
    if args.seed:
        os.environ['PYTHONHASHSEED'] = str(args.seed)

    DEVICE = torch.device(f"npu:{args.local_rank}")
    torch.npu.set_device(DEVICE)
    timer = Timer()

    logging.info(args)
    create_net = create_mobilenetv1_ssd
    config = mobilenetv1_ssd_config

    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    logging.info("Prepare training datasets.")
    datasets = []
    for dataset_path in args.datasets.split(","):
        dataset = VOCDataset(dataset_path, transform=train_transform,
                                target_transform=target_transform)
        label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
        store_labels(label_file, dataset.class_names)
        num_classes = len(dataset.class_names)
        datasets.append(dataset)
    logging.info(f"Stored labels into file {label_file}.")
    train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
    logging.info("Prepare Validation datasets.")
    val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
                                target_transform=target_transform, is_test=True)
    logging.info("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)
    logging.info("Build network.")
    if args.pretrained_ssd:
        net = create_net(args.class_num)
    else: 
        net = create_net(num_classes)
    last_epoch = -1

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    if args.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        net.load(args.resume)
        logging.info("Resume from model successfully")
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    # net.npu(args.local_rank)
    net.to(DEVICE)

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = apex.optimizers.NpuFusedSGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # 添加混合精度支持
    if args.amp:
        net, optimizer = amp.initialize(net, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale_value,combine_grad=True)

    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    if args.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                                     gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    if args.warm_up:
        warm_up_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: epoch / args.warm_up_epochs)    
    # 添加分布式训练支持
    if args.distributed:
        torch.distributed.init_process_group(backend=args.dist_backend,  world_size=args.world_size, rank=args.local_rank)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], broadcast_buffers=False)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                                    num_workers=args.num_workers, pin_memory=False, sampler=train_sampler, drop_last=True)
    for epoch in range(last_epoch + 1, args.num_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            if args.warm_up and epoch < args.warm_up_epochs:
                warm_up_scheduler.step()
            else:
                scheduler.step()
        train(args, train_loader, net, criterion, optimizer,
            device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
        if not args.distributed:
            scheduler.step()

        if (epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1):
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            logging.info(
                    f"Epoch: {epoch}, " +
                    f"Validation Loss: {val_loss:.4f}, " +
                    f"Validation Regression Loss {val_regression_loss:.4f}, " +
                    f"Validation Classification Loss: {val_classification_loss:.4f}"
                )
            if args.local_rank == args.main_rank:
                model_path = os.path.join(args.checkpoint_folder, f"mb1-ssd-Epoch-{epoch}-Loss-{val_loss}.pth")
                if args.distributed:
                    net.module.save(model_path)
                else:
                    net.save(model_path)
                logging.info(f"Saved model {model_path}")


if __name__ == '__main__':
    main()

