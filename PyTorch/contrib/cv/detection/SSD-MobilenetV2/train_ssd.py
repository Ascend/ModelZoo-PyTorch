# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import logging
import pathlib
import sys
import itertools
import apex
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, LambdaLR

from vision.utils.misc import Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from eval_ssd import predicate
from vision.utils.misc import str2bool
from apex import amp

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')

# dataset setting
parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')
parser.add_argument('--data_path', default='')
parser.add_argument('--datasets', default=[], help='Dataset directory path')
parser.add_argument('--validation_dataset', help='Dataset directory path')
parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")

# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net', default='',
                    help='Pretrained base model')
parser.add_argument('--pretrained_ssd', default='', help='Pre-trained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it can be mb1-ssd, mb1-lite-ssd, mb2-ssd-lite, mb3-large-ssd-lite, mb3-small-ssd-lite or vgg16-ssd.")
parser.add_argument('--freeze_base_net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze_net', action='store_true',
                    help="Freeze all the layers except the prediction head.")
parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')

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
# eval params
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--use_2007_metric", type=str2bool, default=True)
parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store evaluation results.")

# distributed setting
parser.add_argument('--distributed', default=False, action='store_true',
                    help='Use multi-processing distributed training to launch ')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--device', default='gpu', type=str, help='npu or gpu')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--device_list', default='0,1,2,3,4,5,6,7', type=str, help='device id list')
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend, nccl for GPU, hccl for NPU')
parser.add_argument('--addr', default='127.0.0.1', type=str, help='master addr')
parser.add_argument('--port', default='29688', type=str, help='master port')

# apex setting
parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--opt_level', default='O2', type=str, help='apex optimize level')
parser.add_argument('--loss_scale_value', default=128.0, type=int, help='static loss scale value')

# learning rate setting
parser.add_argument('--warm_up', default=False, action='store_true', help='use warm_up or not')
parser.add_argument('--warm_up_epochs', default=5, type=int, help='warm up epochs')
parser.add_argument('--stay_lr', default=-1, type=int, help='Epoch with constant learning rate')

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map


def train(loader, net, criterion, optimizer, args, timer, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    timer.start('batch_time')
    timer.start('multi_step_time')
    multi_step = 0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        boxes, labels = boxes.to(args.device), labels.to(args.device)
        optimizer.zero_grad()
        confidence, locations = net.forward(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = (regression_loss + classification_loss)
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        multi_step += 1
        if (i % debug_steps == 0 or i == len(loader) - 1) and (not args.distributed or args.rank == 0):
            avg_loss = running_loss / (i + 1)
            avg_reg_loss = running_regression_loss / (i + 1)
            avg_clf_loss = running_classification_loss / (i + 1)
            multi_step_time = timer.end('multi_step_time')
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"multi_step_time: {multi_step_time:.4f}, " +
                f"step_avg_time: {multi_step_time / multi_step:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}, " +
                f"Average Loss: {avg_loss:.4f}"
            )
            multi_step = 0
            timer.start('multi_step_time')
    if not args.distributed or args.rank == 0:
        batch_time = timer.end('batch_time')
        logging.info(f"Epoch: {epoch}, " +
                     f"batch_time: {batch_time:.4f}, " +
                     f"FPS: {args.batch_size * args.ngpu * len(loader) / batch_time:.4f} ")


def test(loader, net, criterion, args, epoch=-1):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        num += 1
        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence.cpu(), locations.cpu(), labels, boxes)
            loss = regression_loss + classification_loss
        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if not args.distributed or args.rank == 0:
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Regression Loss {running_regression_loss / (i + 1):.4f}, " +
                f"Average Classification Loss: {running_classification_loss / (i + 1):.4f}, " +
                f"Average Loss: {running_loss / (i + 1):.4f}"
            )
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


def main_worker(gpu, timer, args):
    args.gpu = args.process_device_map[gpu]
    print(args.gpu)
    if args.distributed:
        if args.device == 'npu':
            torch.distributed.init_process_group(backend=args.dist_backend,
                                                 world_size=args.ngpu,
                                                 rank=args.rank)
        else:
            torch.distributed.init_process_group(backend=args.dist_backend,
                                                 init_method="env://",
                                                 world_size=args.ngpu,
                                                 rank=args.rank)
    if args.device == 'npu':
        args.device = 'npu:{}'.format(args.gpu)
        print(args.device)
        torch.npu.set_device(args.device)
        logging.info('use NPU, {}'.format(args.device))
    elif args.device == 'gpu':
        args.device = 'cuda:{}'.format(args.gpu)
        torch.backends.cudnn.benchmark = True
        logging.info('use GPU, {}'.format(args.device))

    if args.net == 'vgg16-ssd':
        create_net = create_vgg_ssd
        create_predictor = lambda net: create_vgg_ssd_predictor(net, nms_method=args.nms_method, device=args.device)
        config = vgg_ssd_config
    elif args.net == 'mb1-ssd':
        create_net = create_mobilenetv1_ssd
        create_predictor = lambda net: create_mobilenetv1_ssd_predictor(net, nms_method=args.nms_method,
                                                                        device=args.device)
        config = mobilenetv1_ssd_config
    elif args.net == 'mb1-ssd-lite':
        create_net = create_mobilenetv1_ssd_lite
        create_predictor = lambda net: create_mobilenetv1_ssd_lite_predictor(net, nms_method=args.nms_method,
                                                                             device=args.device)
        config = mobilenetv1_ssd_config
    elif args.net == 'sq-ssd-lite':
        create_net = create_squeezenet_ssd_lite
        create_predictor = lambda net: create_squeezenet_ssd_lite_predictor(net, nms_method=args.nms_method,
                                                                            device=args.device)
        config = squeezenet_ssd_config
    elif args.net == 'mb2-ssd-lite':
        create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult, device=args.device)
        create_predictor = lambda net: create_mobilenetv2_ssd_lite_predictor(net, nms_method=args.nms_method,
                                                                             device=args.device)
        config = mobilenetv1_ssd_config
    elif args.net == 'mb3-large-ssd-lite':
        create_net = lambda num: create_mobilenetv3_large_ssd_lite(num)
        create_predictor = lambda net: create_mobilenetv2_ssd_lite_predictor(net, nms_method=args.nms_method,
                                                                             device=args.device)
        config = mobilenetv1_ssd_config
    elif args.net == 'mb3-small-ssd-lite':
        create_net = lambda num: create_mobilenetv3_small_ssd_lite(num)
        create_predictor = lambda net: create_mobilenetv2_ssd_lite_predictor(net, nms_method=args.nms_method,
                                                                             device=args.device)
        config = mobilenetv1_ssd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    logging.info("Prepare training datasets.")
    datasets = []
    for dataset_path in args.datasets:
        if args.dataset_type == 'voc':
            dataset = VOCDataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform)
            label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
        elif args.dataset_type == 'open_images':
            dataset = OpenImagesDataset(dataset_path,
                                        transform=train_transform, target_transform=target_transform,
                                        dataset_type="train", balance_data=args.balance_data)
            label_file = os.path.join(args.checkpoint_folder, "open-images-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            logging.info(dataset)
            num_classes = len(dataset.class_names)

        else:
            raise ValueError(f"Dataset type {args.dataset_type} is not supported.")
        datasets.append(dataset)
    logging.info(f"Stored labels into file {label_file}.")
    train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(train_dataset)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              args.batch_size,
                              num_workers=args.num_workers,
                              sampler=train_sampler if args.distributed else None,
                              shuffle=False if args.distributed else True)
    logging.info("Prepare Validation datasets.")
    if args.dataset_type == "voc":
        val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
                                 target_transform=target_transform, is_test=True)
    elif args.dataset_type == 'open_images':
        val_dataset = OpenImagesDataset(dataset_path,
                                        transform=test_transform, target_transform=target_transform,
                                        dataset_type="test")
        logging.info(val_dataset)
    logging.info("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)

    logging.info("Build network.")
    net = create_net(num_classes)
    min_loss = -10000.0
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
    net.to(args.device)
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=args.device)
    # npu: NpuFusedSGD
    if 'npu' in args.device:
        optimizer = apex.optimizers.NpuFusedSGD(params, lr=args.lr, momentum=args.momentum,
                                                weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        pretrained_dic = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        net.load_state_dict(pretrained_dic)
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['epoch']
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    if args.amp:
        net, optimizer = amp.initialize(net, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale_value)
    if args.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.rank],
                                                        broadcast_buffers=False if 'npu' in args.device else True)
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

    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, args.num_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.warm_up and epoch < args.warm_up_epochs:
            warm_up_scheduler.step()
        else:
            scheduler.step()
        train(train_loader, net, criterion, optimizer, args, timer,
              debug_steps=args.debug_steps, epoch=epoch)
        if (epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1) and (
                not args.distributed or args.rank == 0):
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, args, epoch)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}, " +
                f"Validation Loss: {val_loss:.4f}"
            )
            model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{val_loss}.pth")
            torch.save({'state_dict': net.state_dict(), 'epoch': epoch, 'optimizer': optimizer.state_dict()},
                       model_path)
            logging.info(f"Saved model {model_path}")

    # 默认只测最后一轮的精度
    predictor = create_predictor(net)
    val_dataset = VOCDataset(args.validation_dataset, is_test=True)
    accuracy = predicate(val_dataset, predictor, args, dataset.class_names)
    logging.info(f'epoch: {epoch}, accuracy: {accuracy}')


if __name__ == '__main__':
    timer = Timer()
    args = parser.parse_args()
    if args.device == 'npu':
        os.environ['MASTER_ADDR'] = args.addr
        os.environ['MASTER_PORT'] = args.port

    logging.info(args)
    args.process_device_map = device_id_to_process_device_map(args.device_list)

    if not os.path.exists(args.eval_dir):
        os.makedirs(args.eval_dir, exist_ok=True)
    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder, exist_ok=True)
    args.datasets = [os.path.join(args.data_path, 'VOC2007_trainval'), os.path.join(args.data_path, 'VOC2012_trainval')]
    args.validation_dataset = os.path.join(args.data_path, 'VOC2007_test')
    if args.distributed:
        args.ngpu = int(os.environ['RANK_SIZE'])
        main_worker(args.rank, timer, args)
    else:
        args.ngpu = 1
        main_worker(args.gpu, timer, args)
