# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pdb
import pprint
import time
import random
import _init_paths
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler
if torch.__version__>="1.8":
    import torch_npu
print(torch.__version__)
import apex
from apex import amp

from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.utils.net_utils import adjust_learning_rate, save_checkpoint, clip_gradient
from roi_data_layer.roibatchLoader import roibatchLoader
from roi_data_layer.roidb import combined_roidb
import constant


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    # parser.add_argument('--arch', dest='arch', default='rfcn', choices=['rcnn', 'rfcn', 'couplenet'])
    parser.add_argument('--arch', dest='arch', default='rfcn', choices=['rcnn', 'rfcn'])
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='res50', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=200, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=2, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--device', default='cpu', type=str,
                        help='choose cpu, npu or cuda')
    parser.add_argument('--npu_id', dest='npu_id',
                        help='npu_id',
                        default='npu:0', type=str)
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--ohem', dest='ohem',
                        help='Use online hard example mining for training',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=4, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # resume trained model
    parser.add_argument('--resume', dest='resume',
                        help='resume checkpoint or not',
                        action="store_true")
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfboard', dest='use_tfboard',
                        help='whether use tensorflow tensorboard',
                        default=False, type=bool)

    # apex
    parser.add_argument('--amp', dest='amp', default=False, action='store_true',
                        help='use amp to train the model')
    parser.add_argument('--loss_scale', dest='loss_scale', default=-1., type=float,
                        help='loss scale using in amp, default -1 means dynamic')
    parser.add_argument('--opt_level', dest='opt_level', default='O2', type=str,
                        help='opt level using in amp, default O2 means FP16')

    # 8p
    parser.add_argument('--addr', default=constant.IP_ADDRESS, type=str, help='master addr')
    parser.add_argument('--device_list', default='0,1,2,3,4,5,6,7', type=str, help='device id list')
    parser.add_argument('--world_size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='node local rank for distributed training')
    parser.add_argument('--dist_backend', default='hccl', type=str,
                            help='distributed backend')
    
    # ETP
    parser.add_argument('--etp_performance_mode', dest='etp_performance_mode', default=False, action='store_true',
                        help='specify trianing steps on ETP performance mode')

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0,batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn.deterministic = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map

def padding_collate(batch):
    im_data_list = []
    im_info_list = []
    num_boxes_list = []
    gt_boxes_list = []

    padding_size = 1344

    for i in range(len(batch)):
        padding_im = torch.zeros(3, padding_size, padding_size)
        size_h = batch[i][0].shape[1]
        size_w = batch[i][0].shape[2]
        padding_im[:, :size_h, :size_w] = batch[i][0]
        im_data_list.append(padding_im)
        im_info_list.append(batch[i][1])
        num_boxes_list.append(batch[i][2])
        gt_boxes_list.append(batch[i][3])

    im_data_tensor = torch.tensor([item.detach().numpy() for item in im_data_list])
    im_info_tensor = torch.tensor([item.detach().numpy() for item in im_info_list])
    num_boxes_tensor = torch.tensor([item.detach().numpy() for item in num_boxes_list])
    gt_boxes_tensor = torch.tensor(gt_boxes_list)
    batch_tensors = (im_data_tensor, im_info_tensor, num_boxes_tensor, gt_boxes_tensor)

    return batch_tensors

def main():
    args = parse_args()

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
    elif args.dataset == "vg":
        # train sizes: train, smalltrain, minitrain
        # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    if args.arch == 'rcnn':
        from model.faster_rcnn.vgg16 import vgg16
        from model.faster_rcnn.resnet import resnet
    elif args.arch == 'rfcn':
        from model.rfcn.resnet_atrous import resnet
    elif args.arch == 'couplenet':
        from model.couplenet.resnet_atrous import resnet

    os.environ['MASTER_ADDR'] = constant.IP_ADDRESS
    os.environ['MASTER_PORT'] = '29688'
    os.environ['WORLD_SIZE'] = '8'

    args.rank = int(args.local_rank)
    npus_per_node = int(os.environ['WORLD_SIZE'])
    calculate_device = f'npu:{str(args.local_rank)}'
    dist.init_process_group(backend=args.dist_backend,  # init_method=cfg.dist_url,
                            world_size=int(os.environ['WORLD_SIZE']), rank=args.rank)
    print(calculate_device)
    torch.npu.set_device(calculate_device)

    if cfg.RNG_SEED is not None:
        seed_everything(cfg.RNG_SEED)
    
    args.batch_size = int(args.batch_size / npus_per_node)
    args.workers = int((args.num_workers + npus_per_node - 1) / npus_per_node)

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)

    if torch.distributed.get_rank() == 0:
        print('{:d} roidb entries'.format(len(roidb)))

    output_dir = os.path.join(args.save_dir, args.arch, args.net, args.dataset)
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

    sampler_batch = sampler(train_size, args.batch_size)

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                             imdb.num_classes, training=True)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=int(os.environ['WORLD_SIZE']), rank=args.rank)
    if torch.distributed.get_rank() == 0:
        print("args.batch_size:", args.batch_size)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=padding_collate,
                                             sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    if args.net == 'vgg16':
        model = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        model = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        model = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        model = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    model.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    #tr_momentum = cfg.TRAIN.MOMENTUM
    #tr_momentum = args.momentum

    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        # optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
        # optimizer = apex.optimizers.NpuFusedSGD(params, momentum=cfg.TRAIN.MOMENTUM)
        optimizer = apex.optimizers.NpuFusedSGD([
            {'params': [param for name, param in model.named_parameters() if param.requires_grad and name[-4:] == 'bias'],
             'lr': lr*(cfg.TRAIN.DOUBLE_BIAS + 1),
             'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0},
            {'params': [param for name, param in model.named_parameters() if param.requires_grad and name[-4:] != 'bias'],
             'lr': lr,
             'weight_decay': cfg.TRAIN.WEIGHT_DECAY}], momentum=cfg.TRAIN.MOMENTUM)
    
    model.npu()

    if args.amp:
        # model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale)
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale, combine_grad=True)
        print("=> Using amp mode.")
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank], broadcast_buffers=False)
    
    if args.resume:
        load_name = os.path.join(output_dir,
                                 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    iters_per_epoch = int(train_size / args.batch_size) / int(os.environ['WORLD_SIZE'])

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        train_sampler.set_epoch(epoch)

        dataset.resize_batch()
        # setting to train mode
        model.train()
        loss_temp = 0
        epoch_start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma
        
        start = time.time()
        data_start = time.time()
        batch_time_sum = 0
        batch_time_mean = 0
        for step, (data0, data1, data2, data3) in enumerate(dataloader):
            if args.etp_performance_mode and step >= 300:
                break
            data_time = (time.time() - data_start) * 1000
            im_data = data0.to(calculate_device, non_blocking=True)
            im_info = data1.to(calculate_device, non_blocking=True)
            gt_boxes = data2.to(calculate_device, non_blocking=True)
            num_boxes = data3.to(calculate_device, non_blocking=True)

            model.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = model(im_data, im_info, gt_boxes, num_boxes)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.item()

            # backward
            optimizer.zero_grad()
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if args.net == "vgg16":
                optimizer.clip_optimizer_grad_norm_fused(10.)
            if args.net == "res101":
                optimizer.clip_optimizer_grad_norm_fused(4.)
            optimizer.step()

            batch_time = (time.time() - start) * 1000
            start = time.time()
            if step > 10:
                batch_time_sum += batch_time
                batch_time_mean = batch_time_sum / (step - 10)
            if step > iters_per_epoch:
                break

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    # TBE不支持long
                    fg_cnt = torch.sum(rois_label.data.int().ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    # TBE不支持long
                    fg_cnt = torch.sum(rois_label.data.int().ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                if torch.distributed.get_rank() == 0:
                    print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                        % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                    # print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
                    if step > 10:
                        print("\t\t\tfg/bg=(%d/%d), batch time: %f, data time: %f, mean batch time: %f, FPS: %f" % (fg_cnt, bg_cnt, batch_time, data_time, batch_time_mean, args.batch_size*npus_per_node/(batch_time_mean/1000)))
                    else:
                        print("\t\t\tfg/bg=(%d/%d), batch time: %f, data time: %f, mean batch time: %f" % (fg_cnt, bg_cnt, batch_time, data_time, batch_time_mean))
                    print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                        % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, step)

                loss_temp = 0
                data_start = time.time()
        
        
        # 每十次保存一次模型
        if epoch % 10 == 0 or epoch == args.max_epochs:
            if args.rank % npus_per_node == 0:
                save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
                save_checkpoint({
                    'session': args.session,
                    'epoch': epoch + 1,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'pooling_mode': cfg.POOLING_MODE,
                    'class_agnostic': args.class_agnostic,
                }, save_name)
                print('save model: {}'.format(save_name))

        # end = time.time()
        # print(end - start)
        torch.npu.synchronize()
        epoch_end = time.time()
        print("epoch_time:", epoch_end - epoch_start)


if __name__ == '__main__':
    main()
    