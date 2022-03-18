#coding=gbk
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
from __future__ import print_function
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pickle
import numpy as np
from utils.timer import Timer
from layers.functions import Detect,PriorBox
from data import BaseTransform
from tqdm import tqdm

import time
import torch
import torch.nn as nn
import random
import shutil
import argparse
from m2det import build_net
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from data import detection_collate
from configs.CC import Config
from utils.core import *
from torch.nn.parallel import DistributedDataParallel as DDP
#amp修改开始
import apex
try:
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
from apex.optimizers import NpuFusedSGD
#amp修改结束

#NPU修改开始
import torch.npu
#NPU修改结束

parser = argparse.ArgumentParser(description='M2Det Training')
parser.add_argument('-c', '--config', default='configs/m2det320_vgg16.py')
parser.add_argument('-d', '--dataset', default='COCO', help='VOC or COCO dataset')
parser.add_argument('--ngpu', default=1, type=int, help='gpus or npus')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-t', '--tensorboard', type=bool, default=False, help='Use tensorborad to show the Loss Graph')
#amp参数开始
parser.add_argument('--amp', default=False, action='store_true',
                        help='use amp to train the model')
parser.add_argument('--opt_level', type=str, default='O2')
parser.add_argument('--loss_scale', type=str, default=128.0)
#amp参数结束
#distributed参数开始
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--dist_rank', default=-1, type=int,
                    help='rank of distributed processes')
parser.add_argument('--multiprocessing_distributed', default=False, action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N NPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
##distributed参数结束
#NPU参数开始
parser.add_argument('--device_list', default='0,1,2,3,4,5,6,7', type=str,
                        help='device id list')
#parser.add_argument('--device', default='npu', type=str, help='npu or cpu')
parser.add_argument('--addr', default='127.0.0.1', type=str,
                        help='master addr')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--device', type=str, default='cpu',
                        help='set which type of device used. Support cuda:0(device_id), npu:0(device_id).')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--FusedSGD', default=False, action='store_true',
                        help='use FusedSGD during trian')
#NPU参数结束
parser.add_argument('--performance', default=False, action='store_true',
                        help='get performace of train model')
parser.add_argument('-m', '--trained_model', default=None, type=str, help='Trained state_dict file path to open')
parser.add_argument('--test', action='store_true', help='to submit a test file')
args = parser.parse_args()


#seed保证重复
def set_seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
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

# 设置hook func
def hook_func(name, module):
    def hook_function(module, inputs, outputs):
        # 请依据使用场景自定义函数
        print(name+' inputs', inputs)
        print(name+' outputs', outputs)
    return hook_function
def run(net, optimizer, criterion, images, priors, targets):
    out = net(images)
    optimizer.zero_grad()
    loss_l, loss_c = criterion(out, priors, targets)
    loss = loss_l + loss_c
    if args.amp:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()
    return loss
        
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = args.process_device_map[gpu]
    print('ngpus_per_node:{}'.format(ngpus_per_node))
    if args.dist_rank % ngpus_per_node == 0:
        logr = set_train_log()
        batch_time = AverageMeter('Time', ':6.3f', start_count_index=6)
        if args.performance:
            perf_time = time.time()
    
    os.environ['LOCAL_DEVICE_ID'] = str(args.gpu)
    if args.gpu is not None:
        if args.device.startswith('cuda'):
            print("[gpu id:",args.gpu,"]","Use GPU: {} for training".format(args.gpu))
        elif args.device.startswith('npu'):
            print("[npu id:",args.gpu,"]","Use NPU: {} for training".format(args.gpu))
        
    if args.distributed:
        if args.dist_url == "env://" and args.dist_rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.dist_rank = args.dist_rank * ngpus_per_node + gpu
        print("rank:",args.dist_rank)
        print("worksize",args.world_size)
        if args.device.startswith('npu'):
            dist.init_process_group(backend=args.dist_backend,  # init_method=args.dist_url,
                                  world_size=args.world_size, rank=args.dist_rank)
        else:
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                  world_size=args.world_size, rank=args.dist_rank)

    calculate_device = 'npu:{}'.format(gpu)#不是args.gpu
    print("device",calculate_device)
    torch.npu.set_device(calculate_device)
    prof_kwargs = {'use_npu': True}#prof
    
    logger = set_logger(args.tensorboard)
    
    global cfg
    cfg = Config.fromfile(args.config)
    net = build_net('train', 
                    size = cfg.model.input_size, # Only 320, 512, 704 and 800 are supported
                    config = cfg.model.m2det_config)
    init_net(net, cfg, args.resume_net) # init the network with pretrained weights or resumed weights
    
    net = net.to(calculate_device)
    #cudnn.benchmark = True

    # 注册正反向hook
    #for name, module in net.named_modules():
    #    module.register_forward_hook(hook_func('[forward]: '+name, module))
    #    module.register_backward_hook(hook_func('[backward]: '+name, module))
    #for name, param in net.named_parameters():
    #    print("[grad]: "+name, param.grad)

    if args.FusedSGD:
        optimizer = NpuFusedSGD(
        net.parameters(),
        lr=cfg.train_cfg.lr[0],
        momentum=cfg.optimizer.momentum,
        weight_decay=cfg.optimizer.weight_decay,
        )
    else:
        optimizer = set_optimizer(net, cfg)
    
    
    #amp修改开始
    if args.amp:
        print("=> use amp, level is", args.opt_level)
        net, optimizer = amp.initialize(net, optimizer,
                                        opt_level=args.opt_level,
                                        loss_scale=args.loss_scale
                                        )
    #amp修改结束
    if args.distributed:
        #net = torch.nn.DataParallel(net)
        #分发模型
        net = DDP(net, device_ids=[args.gpu])
        
    
    criterion = set_criterion(cfg)
    priorbox = PriorBox(anchors(cfg))
    ############## npu modify begin #############
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.to(calculate_device)
    ############## npu modify end #############
    
    net.train()
    epoch = args.resume_epoch
    print_info('===> Loading Dataset...',['yellow','bold'])
    dataset = get_dataloader(cfg, args.dataset, 'train_sets')
    #分发数据
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = None

    epoch_size = len(dataset) // (cfg.train_cfg.per_batch_size * args.ngpu)
    print('epoch_size:{}'.format(epoch_size)) #自己定义
    max_iter = getattr(cfg.train_cfg.step_lr,args.dataset)[-1] * epoch_size
    print('max_iter:{}'.format(max_iter)) #自己定义
    stepvalues = [_*epoch_size for _ in getattr(cfg.train_cfg.step_lr, args.dataset)[:-1]]
    print(stepvalues)
    print_info('===> Training M2Det on ' + args.dataset, ['yellow','bold'])
    step_index = 0
    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0
    logr.info('start training!')
    for iteration in range(start_iter, max_iter):

        if iteration % epoch_size == 0:
            batch_iterator = iter(data.DataLoader(dataset, 
                                                  cfg.train_cfg.per_batch_size, 
                                                  shuffle=(train_sampler is None), 
                                                  num_workers=cfg.train_cfg.num_workers,
                                                  pin_memory=False,
                                                  sampler=train_sampler,
                                                  drop_last=True, 
                                                  collate_fn=detection_collate))
            #设置epoch位置，这应该是个为了同步所做的工作
            if args.distributed:
                train_sampler.set_epoch(epoch)
                
            if epoch % cfg.model.save_eposhs == 0:
                save_checkpoint(net, cfg, final=False, datasetname = args.dataset, epoch=epoch)
            epoch += 1

        #torch.npu.synchronize()
        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, cfg.train_cfg.gamma, epoch, step_index, iteration, epoch_size, cfg)
        data_t0 = time.time()
        images, targets = next(batch_iterator)
        data_time = time.time()-data_t0
        ############## npu modify begin #############
        images = images.to(calculate_device)
        targets = [anno.to(calculate_device) for anno in targets]
        ############## npu modify end #############
        out = net(images)

        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)
        loss = loss_l + loss_c
        write_logger({'loc_loss':loss_l.item(),
                      'conf_loss':loss_c.item(),
                      'loss':loss.item()},logger,iteration,status=args.tensorboard)
        #amp修改开始
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        #amp修改结束
        optimizer.step()
        #torch.npu.synchronize()        
        load_t1 = time.time()
        # measure elapsed time
        batch_time.update(load_t1 - load_t0)
        if iteration == 25:
            # 4. 执行forward+profiling
            with torch.autograd.profiler.profile(**prof_kwargs) as prof:
                run(net, optimizer, criterion, images, priors, targets)
            #print(prof.key_averages().table())
            prof.export_chrome_trace("pytorch_prof_%s.prof" % args.device)
        if args.dist_rank % ngpus_per_node == 0:
            #print_train_log(iteration, cfg.train_cfg.print_epochs,
            #                [time.ctime(),epoch,iteration%epoch_size,epoch_size,iteration,loss_l.item(),loss_c.item(),load_t1-load_t0,lr,loss.item()])
            if iteration % cfg.train_cfg.print_epochs == 0:
                print('Time:{}||Epoch:{}||EpochIter:{}/{}||Iter:{}||Loss_L:{:.4f}||Loss_C:{:.4f}||Batch_Time:{:.4f}||LR:{:.7f}||Loss:{:.4f}'\
                    .format(time.ctime(),epoch,iteration%epoch_size,epoch_size,iteration,loss_l.item(),loss_c.item(),load_t1-load_t0,lr,loss.item()))
            if iteration % cfg.train_cfg.print_epochs == 0:
                logr.info('Time:{}||Epoch:{}||EpochIter:{}/{}||Iter:{}||Loss_L:{:.4f}||Loss_C:{:.4f}||data_Time:{:.4f}||Batch_Time:{:.4f}||LR:{:.7f}'\
                .format(time.ctime(),epoch,iteration%epoch_size,epoch_size,iteration,loss_l.item(),loss_c.item(),data_time,load_t1-load_t0,lr))
            if (iteration+1) % epoch_size == 0:#判断是否完成一个epoch
                if batch_time.avg > 0:
                    FPS = cfg.train_cfg.per_batch_size * args.ngpu / batch_time.avg
                    logr.info('Time:{}||Epoch:{}||Iter:{}||Batch_Time:{:.4f}||batch_time_avg:{:.3f}||Fps:{:.3f}'\
                              .format(time.ctime(),epoch,iteration,load_t1-load_t0,batch_time.avg,FPS))
                    print('Time:{}||Epoch:{}||Iter:{}||Batch_Time:{:.4f}||batch_time_avg:{:.3f}||Fps:{:.3f}'\
                              .format(time.ctime(),epoch,iteration,load_t1-load_t0,batch_time.avg,FPS))
                    batch_time.reset()
        if args.performance:
            perform_time = time.time() - perf_time
            if perform_time > 1800 or iteration > 1000:
                break
    logr.info('finish training!')
    save_checkpoint(net, cfg, final=True, datasetname=args.dataset,epoch=-1)
    validate(cfg, args, priors)

def test_net(save_folder, net, priors, detector, cuda, testset, transform, args, max_per_image=300, thresh=0.005):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    num_images = len(testset)
    print_info('=> Total {} images to test.'.format(num_images),['yellow','bold'])
    num_classes = cfg.model.m2det_config.num_classes
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')
    tot_detect_time, tot_nms_time = 0, 0
    print_info('Begin to evaluate',['yellow','bold'])
    for i in tqdm(range(num_images)):
        img = testset.pull_image(i)
        # step1: CNN detection
        _t['im_detect'].tic()
        boxes, scores = image_forward(img, net, cuda, priors, detector, transform)
        detect_time = _t['im_detect'].toc()
        # step2: Post-process: NMS
        _t['misc'].tic()
        nms_process(num_classes, i, scores, boxes, cfg, thresh, all_boxes, max_per_image)
        nms_time = _t['misc'].toc()

        tot_detect_time += detect_time if i > 0 else 0
        tot_nms_time += nms_time if i > 0 else 0

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    print_info('===> Evaluating detections',['yellow','bold'])
    testset.evaluate_detections(all_boxes, save_folder)
    print_info('Detect time per image: {:.3f}s'.format(tot_detect_time / (num_images - 1)))
    print_info('Nms time per image: {:.3f}s'.format(tot_nms_time / (num_images - 1)))
    print_info('Total time per image: {:.3f}s'.format((tot_detect_time + tot_nms_time) / (num_images - 1)))
    print_info('FPS: {:.3f} fps'.format((num_images - 1) / (tot_detect_time + tot_nms_time)))

def validate(cfg, args, priors):
    print_info('----------------------------------------------------------------------\n'
               '|                       M2Det Evaluation Program                     |\n'
               '----------------------------------------------------------------------', ['yellow','bold'])
    if not os.path.exists(cfg.test_cfg.save_folder):
        os.mkdir(cfg.test_cfg.save_folder)
    anchor_config = anchors(cfg)
    print_info('The Anchor info: \n{}'.format(anchor_config))
    priorbox = PriorBox(anchor_config)
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.npu()
    
    net = build_net('test',
                    size = cfg.model.input_size,
                    config = cfg.model.m2det_config)
    init_net(net, cfg, args.trained_model)
    print_info('===> Finished constructing and loading model',['yellow','bold'])
    net.eval()
    _set = 'eval_sets' if not args.test else 'test_sets'
    testset = get_dataloader(cfg, args.dataset, _set)
    net = net.npu()
    cudnn.benchmark = True

    detector = Detect(cfg.model.m2det_config.num_classes, cfg.loss.bkg_label, anchor_config)
    save_folder = os.path.join(cfg.test_cfg.save_folder, args.dataset)
    _preprocess = BaseTransform(cfg.model.input_size, cfg.model.rgb_means, (2, 0, 1))
    test_net(save_folder, 
             net,
             priors, 
             detector, 
             cfg.test_cfg.cuda, 
             testset, 
             transform = _preprocess,
             args = args, 
             max_per_image = cfg.test_cfg.topk, 
             thresh = cfg.test_cfg.score_threshold,
             )


if __name__ == '__main__':
    print_info('----------------------------------------------------------------------\n'
               '|                       M2Det Training Program                       |\n'
               '----------------------------------------------------------------------',['yellow','bold'])
    if args.seed is not None:
        set_seed_everything(args.seed)
    
    # 8p, 指定训练服务器的ip和端口
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '29501'
    
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    print('addr:{}'.format(args.addr))

    #args.distributed和args.multiprocessing_distributed功能不一样，一个是分布数据，一个是多线程处理  
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    if args.distributed:
        if args.ngpu <= 1:
            print('the number of ngpu <= 1, distributed error!')
            
    # 8p，创建由device_id到process_id的映射参数，获取单节点昇腾910 AI处理器数量。
    args.process_device_map = device_id_to_process_device_map(args.device_list)
    print('process_device_map:{}'.format(args.process_device_map))
    
    if args.device.startswith('npu'):
        ngpus_per_node = len(args.process_device_map)
    else:
        ngpus_per_node = torch.cuda.device_count()
    
    print('{} node found.'.format(ngpus_per_node))
    
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)
    
